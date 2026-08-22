#pragma once

#include <vector>
#include <cmath>
#include "subsystems/ltv/ltv.hpp"
#include "lemlib/pose.hpp"

namespace lemlib {

/**
 * @brief Smooth C^2 Continuous Jerk-Limited Quintic Hermite Spline Generator.
 * Generates optimal, continuous trajectory profiles for LTV-LQR path followers on-the-fly.
 */
class QuinticSplineGenerator {
public:
    struct SplineWaypoints {
        Pose start{0.0f, 0.0f, 0.0f};
        Pose end{0.0f, 0.0f, 0.0f};
        double startVel = 0.0; // m/s
        double endVel = 0.0;   // m/s
        double maxVel = 1.2;   // m/s
        double maxAccel = 2.0; // m/s^2
        double maxJerk = 4.0;  // m/s^3
    };

    /**
     * @brief Generate an S-curve trajectory profile between two poses
     * @param params Waypoint parameters and dynamic physical limits
     * @param dt Time step per trajectory point in seconds (default 0.02s / 50Hz or 0.01s / 100Hz)
     * @return std::vector<State> Formatted trajectory ready for LTVPathFollower
     */
    static std::vector<State> generateTrajectory(const SplineWaypoints& params, double dt = 0.02);

    /**
     * @brief Generate an S-curve trajectory passing through multiple sequential waypoints
     */
    static std::vector<State> generateMultiPointTrajectory(const std::vector<Pose>& waypoints,
                                                           double maxVel = 1.2,
                                                           double maxAccel = 2.0,
                                                           double dt = 0.02);
};

} // namespace lemlib
