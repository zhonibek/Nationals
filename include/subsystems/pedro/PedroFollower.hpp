#pragma once

#include <functional>
#include <vector>
#include "subsystems/pedro/Point.hpp"
#include "subsystems/pedro/BezierCurve.hpp"
#include "subsystems/pedro/PedroPath.hpp"
#include "lemlib/chassis/odom.hpp"
#include "lemlib/util.hpp"

namespace pedro {

/**
 * @brief Configuration parameters for the Pedro Pathing vector follower
 */
struct FollowerConfig {
    float maxVel = 95.0f;           // Max velocity / power [-127, 127]
    float minVel = 16.0f;           // Min velocity to overcome friction during deceleration
    float maxAccel = 50.0f;         // Max acceleration parameter for predictive braking: sqrt(2 * a * s)
    float kP_trans = 4.5f;          // Proportional gain for translational GVF corrective error vector
    float k_centripetal = 0.08f;    // Feedforward gain for centripetal force (v^2 * curvature)
    float kS = 7.0f;                // Static friction feedforward voltage
    float kD_lqr = 0.15f;           // Velocity damping factor
    float exitDistance = 0.8f;      // Target position settle tolerance in inches
    float exitHeadingError = 2.0f;  // Target orientation settle tolerance in degrees
    int settleTimeoutMs = 120;      // Time required within tolerance to declare completion
    float headingKp = 3.2f;         // Decoupled heading PID Proportional gain
    float headingKi = 0.02f;        // Decoupled heading PID Integral gain
    float headingKd = 2.0f;         // Decoupled heading PID Derivative gain
};

/**
 * @brief Reactive Vector Field Follower (Guiding Vector Field / GVF)
 * based on Pedro Pathing (FTC) adapted specifically for VEX V5 Holonomic robots.
 * 
 * Features:
 * - Reactive Vector Field: Robot continuously corrects back to the curve if displaced or bumped.
 * - Decoupled Heading: Holonomic 3-DOF control allows translating along curves while facing any angle.
 * - Predictive Braking: Smooth, overshoot-free deceleration using sqrt(2 * a * s_remaining).
 * - Centripetal Compensation: Cancels outward drift on sharp spline curves.
 */
class PedroFollower {
public:
    explicit PedroFollower(FollowerConfig config = FollowerConfig());

    void setConfig(const FollowerConfig& cfg) { this->config = cfg; }
    const FollowerConfig& getConfig() const { return config; }

    /**
     * @brief Follow a complete PedroPath with decoupled heading
     * 
     * @param path Trajectory with Bézier segments and heading specifications
     * @param timeoutMs Maximum allowed execution time in ms
     * @param driveFn Function to send (forward, strafe, turn) motor commands
     * @param brakeFn Function to stop and brake motors on completion
     */
    void follow(const PedroPath& path,
                int timeoutMs,
                std::function<void(int forward, int strafe, int turn)> driveFn,
                std::function<void()> brakeFn);

    /**
     * @brief Follow a single Bézier curve
     */
    void follow(const BezierCurve& curve,
                HeadingMode headingMode,
                float targetHeadingDeg,
                int timeoutMs,
                std::function<void(int forward, int strafe, int turn)> driveFn,
                std::function<void()> brakeFn);

    /**
     * @brief Cancel active motion
     */
    void cancel() { cancelled = true; }

private:
    FollowerConfig config;
    bool cancelled = false;
};

} // namespace pedro
