#pragma once

#include <vector>
#include <atomic>
#include <random>
#include "pros/rtos.hpp"
#include "pros/distance.hpp"
#include "lemlib/pose.hpp"
#include "lemlib/chassis/chassis.hpp"

namespace lemlib {

struct Particle {
    double x;      // inches
    double y;      // inches
    double theta;  // radians
    double weight;
};

/**
 * @brief Fixed-size Monte Carlo Localization (MCL) Particle Filter.
 * Estimates global 2D robot pose using distance sensor raycasts against known field walls.
 */
class MCL {
public:
    MCL(Chassis& chassis,
        const std::vector<std::pair<pros::Distance*, double>>& sensorsWithMountAngles,
        int numParticles = 2000,
        double fieldSizeInches = 144.0);

    ~MCL();

    void init(const Pose& initialPose, double posStdDev = 2.0, double headingStdDevDeg = 5.0);

    void startTask(uint32_t periodMs = 33);
    void stopTask();

    /**
     * @brief Execute one step of MCL (motion update, raycast measurement update, resample)
     */
    void update();

    Pose getEstimatedPose() const;
    bool isConverged() const;
    void setSeed(uint32_t seed);
    // Per-sensor mounting offset: right and forward inches. Angles are clockwise radians.
    void setSensorOffsets(const std::vector<std::pair<double,double>>& offsets);

private:
    Chassis& chassis;
    std::vector<std::pair<pros::Distance*, double>> sensors;
    int particleCount;
    double fieldSize;

    std::vector<Particle> particles;
    Pose lastOdomPose = Pose(0, 0, 0);
    Pose estimatedPose = Pose(0, 0, 0);
    std::atomic<bool> converged{false};
    bool initialized=false;
    unsigned convergenceSamples=0;
    std::vector<std::pair<double,double>> offsets;
    pros::Mutex lifecycleMutex;
    mutable pros::Mutex mclMutex;

    pros::Task* task = nullptr;
    std::atomic<bool> running{false};
    uint32_t checkPeriodMs = 33;

    std::mt19937 rng;

    static void task_fn(void* param);

    // Raycast distance from (px, py, angle) to square field boundary [-fieldSize/2, +fieldSize/2]
    double raycastField(double px, double py, double rayAngleRad) const;
};

} // namespace lemlib
