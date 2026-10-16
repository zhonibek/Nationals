#pragma once
#include "subsystems/control/Cascade.hpp"
#include "subsystems/control/GVTReference.hpp"
#include "lemlib/safety.hpp"
#include <functional>

namespace nationals {

// Hardware boundary. Reference generators never receive ground truth or motors.
Config configuredCascade();
Feedback readFeedback(pros::MotorGroup& left, pros::MotorGroup& right, double wheelDiameter, double gearRatio, uint32_t expectedEpoch);

/**
 * @brief Follow a time-parameterized reference using LTV-LQR + PID Cascade.
 */
lemlib::MotionResult followReference(pros::MotorGroup& left, pros::MotorGroup& right,
    double wheelDiameter, double gearRatio, const Config& config,
    double duration, double timeout, const std::function<Reference(double)>& reference,
    const std::function<bool()>& cancelled);

/**
 * @brief Follow a state-dependent reference using LTV-LQR + PID Cascade.
 */
lemlib::MotionResult followReference(pros::MotorGroup& left, pros::MotorGroup& right,
    double wheelDiameter, double gearRatio, const Config& config,
    double duration, double timeout, const std::function<Reference(double, const Feedback&)>& reference,
    const std::function<bool()>& cancelled);

/**
 * @brief Follow a GVT (Guiding Vector Tracking) reference using LTV-LQR + PID Cascade.
 * Unites Pedro Pathing vector field path following with optimal Riccati state feedback and 4-wheel PIDf.
 */
lemlib::MotionResult followGVT(pros::MotorGroup& left, pros::MotorGroup& right,
    double wheelDiameter, double gearRatio, const Config& config,
    GVTReference& gvt, double timeout,
    const std::function<bool()>& cancelled);

/**
 * @brief Helper to follow a single Bézier curve with GVT + LTV-LQR + PID Cascade.
 */
lemlib::MotionResult followGVT(pros::MotorGroup& left, pros::MotorGroup& right,
    double wheelDiameter, double gearRatio, const Config& config,
    const pedro::BezierCurve& curve, float startHeadingDeg, float endHeadingDeg,
    double maxSpeed, double maxAccel, double timeout,
    const std::function<bool()>& cancelled);

} // namespace nationals
