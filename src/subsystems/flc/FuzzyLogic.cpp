#include "subsystems/flc/FuzzyLogic.hpp"

namespace lemlib {

FuzzyLogicController::FuzzyLogicController(double maxExpectedError, double maxExpectedRate)
    : maxError(maxExpectedError > 1e-3 ? maxExpectedError : 1.0),
      maxRate(maxExpectedRate > 1e-3 ? maxExpectedRate : 1.0) {}

void FuzzyLogicController::setNormalization(double maxExpectedError, double maxExpectedRate) {
    if (maxExpectedError > 1e-3) maxError = maxExpectedError;
    if (maxExpectedRate > 1e-3) maxRate = maxExpectedRate;
}

double FuzzyLogicController::trimf(double x, double a, double b, double c) {
    if (x <= a || x >= c) return 0.0;
    if (x == b) return 1.0;
    if (x < b) return (x - a) / (b - a);
    return (c - x) / (c - b);
}

double FuzzyLogicController::trapmf(double x, double a, double b, double c, double d) {
    if (x <= a || x >= d) return 0.0;
    if (x >= b && x <= c) return 1.0;
    if (x < b) return (x - a) / (b - a);
    return (d - x) / (d - c);
}

void FuzzyLogicController::computeMultipliers(double error, double rate, double& outKpMult, double& outKvMult) const {
    if (std::isnan(error) || std::isnan(rate) || std::isinf(error) || std::isinf(rate)) {
        outKpMult = 1.0;
        outKvMult = 1.0;
        return;
    }

    // Normalize inputs to [-1.0, +1.0]
    double eNorm = std::clamp(error / maxError, -1.0, 1.0);
    double rNorm = std::clamp(rate / maxRate, -1.0, 1.0);

    // 1. Membership evaluation for Error (E)
    double e_NB = trapmf(eNorm, -2.0, -1.0, -0.6, -0.2);
    double e_NS = trimf(eNorm, -0.6, -0.25, 0.0);
    double e_ZE = trimf(eNorm, -0.25, 0.0, 0.25);
    double e_PS = trimf(eNorm, 0.0, 0.25, 0.6);
    double e_PB = trapmf(eNorm, 0.2, 0.6, 1.0, 2.0);

    // 2. Membership evaluation for Rate/Velocity (R)
    double r_NB = trapmf(rNorm, -2.0, -1.0, -0.6, -0.2);
    double r_NS = trimf(rNorm, -0.6, -0.25, 0.0);
    double r_ZE = trimf(rNorm, -0.25, 0.0, 0.25);
    double r_PS = trimf(rNorm, 0.0, 0.25, 0.6);
    double r_PB = trapmf(rNorm, 0.2, 0.6, 1.0, 2.0);

    // Takagi-Sugeno Singleton Rule Outputs:
    // If Error is Big -> High Kp (1.30), Moderate Kv (1.0)
    // If Error is Zero & Rate is High -> High Kv (1.50) to prevent overshoot
    // If Error is Small & Rate is Small -> Normal Kp (1.0), Stable Kv (1.10)
    
    double numKp = 0.0;
    double numKv = 0.0;
    double den = 0.0;

    auto applyRule = [&](double w, double kp_val, double kv_val) {
        if (w > 0.001) {
            numKp += w * kp_val;
            numKv += w * kv_val;
            den += w;
        }
    };

    // Error Big (Positive or Negative) -> Accelerate fast
    applyRule(std::max(e_NB, e_PB), 1.35, 0.90);

    // Error Medium, moving fast towards target -> Brake hard
    applyRule(e_PS * r_NB, 1.10, 1.45);
    applyRule(e_NS * r_PB, 1.10, 1.45);

    // Error Medium, moving away from target -> Push back
    applyRule(e_PS * r_PB, 1.25, 1.20);
    applyRule(e_NS * r_NB, 1.25, 1.20);

    // Error Zero, High Velocity (Overshoot danger) -> Maximum damping
    applyRule(e_ZE * std::max(r_NB, r_PB), 0.90, 1.60);

    // Error Zero, Low Velocity (Settling) -> Smooth holding
    applyRule(e_ZE * e_ZE, 1.00, 1.05);

    if (den > 1e-4) {
        outKpMult = std::clamp(numKp / den, 0.70, 1.40);
        outKvMult = std::clamp(numKv / den, 0.70, 1.60);
    } else {
        outKpMult = 1.0;
        outKvMult = 1.0;
    }
}

} // namespace lemlib
