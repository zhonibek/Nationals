#include <cmath>
#include "lemlib/lqr.hpp"
#include "lemlib/util.hpp"

namespace lemlib {

LQR::LQR(float kP, float kV, float kA, float kI, float windupRange, bool signFlipReset)
    : kP(kP),
      kV(kV),
      kA(kA),
      kI(kI),
      windupRange(windupRange),
      signFlipReset(signFlipReset) {}

float LQR::update(float error, float velocity, float accel, float dt) {
    if (dt <= 0) dt = 0.01f;

    // update integral
    integral += error;
    if (sgn(error) != sgn(prevError) && signFlipReset) integral = 0;
    if (std::fabs(error) > windupRange && windupRange != 0) integral = 0;
    prevError = error;

    // 1-step predictive lookahead using instantaneous acceleration
    // (Only used for 1-step prediction; never integrated over time to prevent drift)
    float predictedVelocity = velocity + accel * dt;
    float predictedError = error - (velocity * dt) - (0.5f * accel * dt * dt);

    // full state feedback control law:
    // u = kP * e_pred - kV * v_pred - kA * a + kI * integral
    float output = (kP * predictedError) - (kV * predictedVelocity) - (kA * accel) + (kI * integral);

    return output;
}

float LQR::update(float error, float dt) {
    if (dt <= 0) dt = 0.01f;

    // If external velocity is not provided, estimate from change in position error
    float derivative = (prevError - error) / dt; // rate of closing the error
    filteredVelocity = ema(derivative, filteredVelocity, 0.75f);

    return update(error, filteredVelocity, 0, dt);
}

void LQR::reset() {
    integral = 0;
    prevError = 0;
    filteredVelocity = 0;
}

void LQR::setGains(float kP, float kV, float kA, float kI) {
    this->kP = kP;
    this->kV = kV;
    this->kA = kA;
    this->kI = kI;
}

} // namespace lemlib
