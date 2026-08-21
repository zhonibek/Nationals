#include "subsystems/mcl/MCL.hpp"
#include "lemlib/util.hpp"
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
      particleCount(numParticles),
      fieldSize(fieldSizeInches),
      rng(std::random_device{}()) {
    particles.resize(particleCount);
}

MCL::~MCL() {
    stopTask();
}

void MCL::init(const Pose& initialPose, double posStdDev, double headingStdDevDeg) {
    mclMutex.take(TIMEOUT_MAX);
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
    if (running) return;
    checkPeriodMs = periodMs;
    running = true;
    task = new pros::Task(task_fn, this, "MCLTask");
}

void MCL::stopTask() {
    if (!running) return;
    running = false;
    if (task) {
        task->remove();
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
    Pose currentOdom = chassis.getPose(true);
    double deltaX = currentOdom.x - lastOdomPose.x;
    double deltaY = currentOdom.y - lastOdomPose.y;
    double deltaTheta = currentOdom.theta - lastOdomPose.theta;
    lastOdomPose = currentOdom;

    // 1. Motion Model Update with process noise
    std::normal_distribution<double> noiseX(0.0, 0.15);
    std::normal_distribution<double> noiseY(0.0, 0.15);
    std::normal_distribution<double> noiseTheta(0.0, 0.01);

    double half = fieldSize / 2.0;

    for (auto& p : particles) {
        p.x += deltaX + noiseX(rng);
        p.y += deltaY + noiseY(rng);
        p.theta += deltaTheta + noiseTheta(rng);

        // Clamp to field bounds
        p.x = std::clamp(p.x, -half, half);
        p.y = std::clamp(p.y, -half, half);
    }

    // 2. Measurement Model (Sensor Likelihood)
    struct ValidReading {
        double actualDist;
        double mountAngle;
    };
    std::vector<ValidReading> validReadings;

    for (const auto& [sensor, mountAngle] : sensors) {
        if (!sensor) continue;
        int32_t mm = sensor->get();
        if (mm > 0 && mm != PROS_ERR) {
            double distInches = mm / 25.4;
            if (distInches >= 3.0 && distInches <= 60.0) {
                validReadings.push_back({ distInches, mountAngle });
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
                double expectedDist = raycastField(p.x, p.y, rayAngle);
                if (expectedDist > 0) {
                    double error = r.actualDist - expectedDist;
                    logLikelihood -= (error * error) / twoSigmaSq;
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
    mclMutex.give();
}

Pose MCL::getEstimatedPose() const {
    mclMutex.take(TIMEOUT_MAX);
    Pose p = estimatedPose;
    mclMutex.give();
    return p;
}

bool MCL::isConverged() const {
    return converged;
}

} // namespace lemlib
