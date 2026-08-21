#pragma once

#include <vector>
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
 * @brief Adaptive Monte Carlo Localization (AMCL) Particle Filter.
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

private:
    Chassis& chassis;
    std::vector<std::pair<pros::Distance*, double>> sensors;
    int particleCount;
    double fieldSize;

    std::vector<Particle> particles;
    Pose lastOdomPose = Pose(0, 0, 0);
    Pose estimatedPose = Pose(0, 0, 0);
    bool converged = false;

    pros::Task* task = nullptr;
    bool running = false;
    uint32_t checkPeriodMs = 33;

    std::mt19937 rng;

    static void task_fn(void* param);

    // Raycast distance from (px, py, angle) to square field boundary [-fieldSize/2, +fieldSize/2]
    double raycastField(double px, double py, double rayAngleRad) const;
};

} // namespace lemlib
