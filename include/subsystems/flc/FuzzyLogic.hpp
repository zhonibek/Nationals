#pragma once

#include <cmath>
#include <algorithm>

namespace lemlib {

/**
 * @brief Linguistic fuzzy terms for Mamdani/Takagi-Sugeno inference.
 */
enum class FuzzySet {
    NB, // Negative Big
    NS, // Negative Small
    ZE, // Zero
    PS, // Positive Small
    PB  // Positive Big
};

/**
 * @brief Adaptive Fuzzy Logic Controller (FLC).
 * Dynamically adjusts proportional (kP) and damping (kV / kD) gain multipliers
 * based on the instantaneous magnitude of position error and rate of change.
 */
class FuzzyLogicController {
public:
    /**
     * @brief Construct a new Fuzzy Logic Controller
     * @param maxExpectedError Maximum expected error for normalization (e.g., 24 inches or 90 deg)
     * @param maxExpectedRate Maximum expected derivative rate (e.g., 40 in/s or 180 deg/s)
     */
    FuzzyLogicController(double maxExpectedError = 24.0, double maxExpectedRate = 40.0);

    /**
     * @brief Compute adaptive multipliers for kP and kV/kD.
     * @param error Instantaneous error (target - current)
     * @param rate Instantaneous rate of change (derivative of error or velocity)
     * @param[out] outKpMult Multiplier for proportional gain (typically 0.7 to 1.4)
     * @param[out] outKvMult Multiplier for damping/derivative gain (typically 0.7 to 1.6)
     */
    void computeMultipliers(double error, double rate, double& outKpMult, double& outKvMult) const;

    void setNormalization(double maxExpectedError, double maxExpectedRate);

private:
    double maxError;
    double maxRate;

    // Triangular membership function evaluation
    static double trimf(double x, double a, double b, double c);
    
    // Trapezoidal membership function evaluation
    static double trapmf(double x, double a, double b, double c, double d);
};

} // namespace lemlib
