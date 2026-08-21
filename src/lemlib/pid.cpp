#include "pid.hpp"
#include "util.hpp"

namespace lemlib {
PID::PID(float kP, float kI, float kD, float windupRange, bool signFlipReset)
    : kP(kP),
      kI(kI),
      kD(kD),
      windupRange(windupRange),
      signFlipReset(signFlipReset) {}

float PID::update(const float error, float dt) {
    if (dt <= 0) dt = 0.01f;

    // calculate integral with anti-windup clamping
    if (windupRange == 0 || std::fabs(error) <= windupRange) {
        integral += error * dt;
        integral = std::clamp(integral, -30.0f, 30.0f);
    } else {
        integral = 0;
    }

    if (sgn(error) != sgn(prevError) && signFlipReset) {
        integral = 0;
    }

    // calculate filtered derivative to prevent high frequency noise vibration
    if (isFirstStep) {
        prevError = error;
        filteredDerivative = 0.0f;
        isFirstStep = false;
    } else {
        float rawDerivative = (error - prevError) / dt;
        filteredDerivative = ema(rawDerivative, filteredDerivative, 0.75f);
        prevError = error;
    }

    // calculate output (scaled appropriately for motor output)
    return (error * kP) + (integral * kI) + (filteredDerivative * kD * 0.01f);
}

void PID::reset() {
    integral = 0;
    prevError = 0;
    filteredDerivative = 0;
    isFirstStep = true;
}
} // namespace lemlib