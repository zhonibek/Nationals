#pragma once
#include "subsystems/control/Cascade.hpp"
#include "lemlib/safety.hpp"
#include <functional>

namespace nationals {
// Hardware boundary. Reference generators never receive ground truth or motors.
Config configuredCascade();
Feedback readFeedback(pros::MotorGroup& left,pros::MotorGroup& right,double wheelDiameter,double gearRatio,uint32_t expectedEpoch);
lemlib::MotionResult followReference(pros::MotorGroup& left,pros::MotorGroup& right,
    double wheelDiameter,double gearRatio,const Config& config,
    double duration,double timeout,const std::function<Reference(double)>& reference,
    const std::function<bool()>& cancelled);
}
