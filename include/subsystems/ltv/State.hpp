#pragma once

#include <cmath>

namespace lemlib {

/**
 * @brief Trajectory state along a planned path.
 */
struct State {
    double x = 0.0;          // Global X coordinate in meters
    double y = 0.0;          // Global Y coordinate in meters
    double heading = 0.0;    // Target heading in radians
    double linear_vel = 0.0; // Target linear velocity in m/s
    double angular_vel = 0.0;// Target angular velocity in rad/s
};

} // namespace lemlib
