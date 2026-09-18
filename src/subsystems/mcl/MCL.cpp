#include "subsystems/mcl/MCL.hpp"
#include "lemlib/util.hpp"
#include "lemlib/chassis/odom.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace lemlib {

MCL::MCL(Chassis& chassis,
         const std::vector<std::pair<pros::Distance*, double>>& sensorsWithMountAngles,
         int numParticles,
         double fieldSizeInches)
    : chassis(chassis),
      sensors(sensorsWithMountAngles),
      particleCount(std::clamp(numParticles,32,10000)),
      fieldSize(std::isfinite(fieldSizeInches)&&fieldSizeInches>0?fieldSizeInches:144),
      rng(0x5EEDu) {
    particles.resize(particleCount);
    offsets.resize(sensors.size(),{0,0});
}

MCL::~MCL() {
    stopTask();
}

void MCL::init(const Pose& initialPose, double posStdDev, double headingStdDevDeg) {
    if(!std::isfinite(initialPose.x)||!std::isfinite(initialPose.y)||!std::isfinite(initialPose.theta)||!std::isfinite(posStdDev)||posStdDev<0||!std::isfinite(headingStdDevDeg)||headingStdDevDeg<0) return;
    mclMutex.take(TIMEOUT_MAX);
    initialized=true;converged=false;convergenceSamples=0;
    std::normal_distribution<double> distX(initialPose.x, posStdDev);
    std::normal_distribution<double> distY(initialPose.y, posStdDev);
    std::normal_distribution<double> distTheta(degToRad(initialPose.theta), degToRad(headingStdDevDeg));

    for (int i = 0; i < particleCount; ++i) {
        particles[i].x = distX(rng);
        particles[i].y = distY(rng);
        particles[i].theta = distTheta(rng);
        particles[i].weight = 1.0 / particleCount;
    }

    lastOdomPose = chassis.getPose(true);
    estimatedPose = initialPose;
    mclMutex.give();
}

void MCL::startTask(uint32_t periodMs) {
    Lock guard(lifecycleMutex);
    if (running) return;
    checkPeriodMs = std::clamp<uint32_t>(periodMs,10u,1000u);
    running = true;
    task = new pros::Task(task_fn, this, "MCLTask");
}

void MCL::stopTask() {
    Lock guard(lifecycleMutex);
    running = false;
    if (task) {
        task->join();
        delete task;
        task = nullptr;
    }
}

void MCL::task_fn(void* param) {
    auto* self = static_cast<MCL*>(param);
    while (self->running) {
        self->update();
        pros::delay(self->checkPeriodMs);
    }
}

double MCL::raycastField(double px, double py, double rayAngleRad) const {
    double half = fieldSize / 2.0;
    double cosA = std::cos(rayAngleRad);
    double sinA = std::sin(rayAngleRad);

    double minDist = 1e9;

    // Intersect with East Wall (x = +half)
    if (sinA > 1e-4) {
        double d = (half - px) / sinA;
        double hitY = py + d * cosA;
        if (d > 0 && hitY >= -half && hitY <= half) minDist = std::min(minDist, d);
    }
    // Intersect with West Wall (x = -half)
    if (sinA < -1e-4) {
        double d = (-half - px) / sinA;
        double hitY = py + d * cosA;
        if (d > 0 && hitY >= -half && hitY <= half) minDist = std::min(minDist, d);
    }
    // Intersect with North Wall (y = +half)
    if (cosA > 1e-4) {
        double d = (half - py) / cosA;
        double hitX = px + d * sinA;
        if (d > 0 && hitX >= -half && hitX <= half) minDist = std::min(minDist, d);
    }
    // Intersect with South Wall (y = -half)
    if (cosA < -1e-4) {
        double d = (-half - py) / cosA;
        double hitX = px + d * sinA;
        if (d > 0 && hitX >= -half && hitX <= half) minDist = std::min(minDist, d);
    }

    return (minDist < 1e8) ? minDist : -1.0;
}

void MCL::update() {
    mclMutex.take(TIMEOUT_MAX);
    if(!initialized || !getOdomStatus().valid) {converged=false; mclMutex.give(); return;}
    Pose currentOdom = chassis.getPose(true);
    double deltaX = currentOdom.x - lastOdomPose.x;
    double deltaY = currentOdom.y - lastOdomPose.y;
    double deltaTheta = currentOdom.theta - lastOdomPose.theta;
    const double mid=lastOdomPose.theta+deltaTheta/2;
    const double forward=deltaX*std::sin(mid)+deltaY*std::cos(mid);
    const double right=deltaX*std::cos(mid)-deltaY*std::sin(mid);
    lastOdomPose = currentOdom;

    // 1. Motion Model Update with process noise
    std::normal_distribution<double> noiseX(0.0, 0.15);
    std::normal_distribution<double> noiseY(0.0, 0.15);
    std::normal_distribution<double> noiseTheta(0.0, 0.01);

    double half = fieldSize / 2.0;

    for (auto& p : particles) {
        const double h=p.theta+deltaTheta/2;
        p.x += forward*std::sin(h)+right*std::cos(h)+noiseX(rng);
        p.y += forward*std::cos(h)-right*std::sin(h)+noiseY(rng);
        p.theta += deltaTheta + noiseTheta(rng);

        // Clamp to field bounds
        p.x = std::clamp(p.x, -half, half);
        p.y = std::clamp(p.y, -half, half);
    }

    // 2. Measurement Model (Sensor Likelihood)
    struct ValidReading {
        double actualDist;
        double mountAngle;
        double right,forward;
    };
    std::vector<ValidReading> validReadings;

    for(size_t sensorIndex=0;sensorIndex<sensors.size();++sensorIndex) {
        const auto& [sensor,mountAngle]=sensors[sensorIndex];
        if (!sensor) continue;
        int32_t mm = sensor->get();
        if (mm > 0 && mm != PROS_ERR) {
            double distInches = mm / 25.4;
            if (distInches >= 3.0 && distInches <= 60.0) {
                const auto [right,forward]=offsets[sensorIndex];
                const double sx=currentOdom.x+right*std::cos(currentOdom.theta)+forward*std::sin(currentOdom.theta);
                const double sy=currentOdom.y-right*std::sin(currentOdom.theta)+forward*std::cos(currentOdom.theta);
                const double predicted=raycastField(sx,sy,currentOdom.theta+mountAngle);
                // Wall-only model: reject likely occluders instead of snapping to field objects.
                if(predicted>0 && std::abs(predicted-distInches)<12) validReadings.push_back({distInches,mountAngle,right,forward});
            }
        }
    }

    if (!validReadings.empty()) {
        double totalWeight = 0.0;
        constexpr double sigma = 2.0; // sensor standard deviation in inches
        constexpr double twoSigmaSq = 2.0 * sigma * sigma;

        for (auto& p : particles) {
            double logLikelihood = 0.0;
            for (const auto& r : validReadings) {
                double rayAngle = p.theta + r.mountAngle;
                double expectedDist = raycastField(p.x+r.right*std::cos(p.theta)+r.forward*std::sin(p.theta),p.y-r.right*std::sin(p.theta)+r.forward*std::cos(p.theta),rayAngle);
                if (expectedDist > 0) {
                    double error = r.actualDist - expectedDist;
                    logLikelihood += std::log(0.95*std::exp(-(error*error)/twoSigmaSq)+0.05/fieldSize);
                } else {
                    logLikelihood -= 10.0;
                }
            }
            p.weight = std::exp(std::max(logLikelihood, -50.0));
            totalWeight += p.weight;
        }

        // Normalize weights
        if (totalWeight > 1e-9) {
            for (auto& p : particles) {
                p.weight /= totalWeight;
            }

            // 3. Low-variance Resampling
            std::vector<Particle> newParticles;
            newParticles.reserve(particleCount);

            std::uniform_real_distribution<double> dist01(0.0, 1.0 / particleCount);
            double r = dist01(rng);
            double c = particles[0].weight;
            int idx = 0;

            for (int m = 0; m < particleCount; ++m) {
                double u = r + (double)m / particleCount;
                while (u > c && idx < particleCount - 1) {
                    idx++;
                    c += particles[idx].weight;
                }
                newParticles.push_back(particles[idx]);
                newParticles.back().weight = 1.0 / particleCount;
            }
            particles = std::move(newParticles);
        }
    }

    // 4. Compute Estimated Pose (Mean)
    double meanX = 0.0;
    double meanY = 0.0;
    double sinSum = 0.0;
    double cosSum = 0.0;

    for (const auto& p : particles) {
        meanX += p.x;
        meanY += p.y;
        sinSum += std::sin(p.theta);
        cosSum += std::cos(p.theta);
    }

    meanX /= particleCount;
    meanY /= particleCount;
    double meanTheta = std::atan2(sinSum, cosSum);

    estimatedPose = Pose(meanX, meanY, radToDeg(meanTheta));
    double variance=0;
    for(const auto& p:particles) variance+=(p.x-meanX)*(p.x-meanX)+(p.y-meanY)*(p.y-meanY);
    variance/=particleCount;
    const double concentration=std::hypot(sinSum,cosSum)/particleCount;
    if(validReadings.size()>=2 && variance<4 && concentration>0.996) ++convergenceSamples;
    else convergenceSamples=0;
    converged=convergenceSamples>=10;
    mclMutex.give();
}

Pose MCL::getEstimatedPose() const {
    mclMutex.take(TIMEOUT_MAX);
    Pose p = estimatedPose;
    mclMutex.give();
    return p;
}

void MCL::setSeed(uint32_t seed) {Lock guard(mclMutex);rng.seed(seed);}
void MCL::setSensorOffsets(const std::vector<std::pair<double,double>>& input) {
    if(input.size()!=sensors.size()) return;
    for(auto [x,y]:input) if(!std::isfinite(x)||!std::isfinite(y)) return;
    Lock guard(mclMutex);offsets=input;converged=false;convergenceSamples=0;
}
bool MCL::isConverged() const {
    return converged;
}

} // namespace lemlib
