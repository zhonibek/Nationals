#pragma once

namespace lemlib {

/**
 * @brief Enum class for selecting the active motion controller type
 */
enum class MotionControllerType {
    PID, /**< Traditional Proportional-Integral-Derivative controller */
    LQR  /**< Linear Quadratic Regulator with state feedback & predictive acceleration */
};

/**
 * @brief Class containing constants and configuration for an LQR controller
 */
class LQRSettings {
    public:
        /**
         * @brief LQRSettings constructor
         *
         * @param kP position/angle state error gain (analogous to proportional gain)
         * @param kV velocity state feedback gain (damping term, counters momentum)
         * @param kA predictive acceleration gain (uses instantaneous IMU accel for 1-step lookahead)
         * @param kI integral gain for eliminating steady-state offset (0 to disable)
         * @param windupRange integral anti-windup range
         * @param smallError small error threshold
         * @param smallErrorTimeout small error timeout in milliseconds
         * @param largeError large error threshold
         * @param largeErrorTimeout large error timeout in milliseconds
         * @param slew maximum acceleration rate limit (0 to disable)
         * @param useImuPrediction whether to use IMU acceleration for predictive damping (default true)
         */
        LQRSettings(float kP, float kV, float kA, float kI, float windupRange, float smallError,
                    float smallErrorTimeout, float largeError, float largeErrorTimeout, float slew,
                    bool useImuPrediction = true)
            : kP(kP),
              kV(kV),
              kA(kA),
              kI(kI),
              windupRange(windupRange),
              smallError(smallError),
              smallErrorTimeout(smallErrorTimeout),
              largeError(largeError),
              largeErrorTimeout(largeErrorTimeout),
              slew(slew),
              useImuPrediction(useImuPrediction) {}

        float kP;
        float kV;
        float kA;
        float kI;
        float windupRange;
        float smallError;
        float smallErrorTimeout;
        float largeError;
        float largeErrorTimeout;
        float slew;
        bool useImuPrediction;
};

/**
 * @brief Linear Quadratic Regulator (LQR) Controller with predictive state estimation
 *
 * Implements optimal full-state feedback:
 * u = kP * error - kV * velocity - kA * acceleration + kI * integral
 *
 * Accelerometer data is used strictly for a 1-step predictive correction to dampen lag
 * and external disturbances without integrating over time (guaranteeing zero drift).
 */
class LQR {
    public:
        /**
         * @brief Construct a new LQR controller
         *
         * @param kP position / primary state error gain
         * @param kV velocity state feedback damping gain
         * @param kA predictive acceleration gain (0 to disable)
         * @param kI integral gain (0 to disable)
         * @param windupRange integral anti-windup range
         * @param signFlipReset whether to reset integral when error sign changes
         */
        LQR(float kP, float kV, float kA = 0, float kI = 0, float windupRange = 0, bool signFlipReset = false);

        /**
         * @brief Update the LQR controller with full measured state and predictive acceleration
         *
         * @param error target minus current position/angle (e)
         * @param velocity current measured velocity / rate of change (v)
         * @param accel instantaneous measured acceleration (used strictly for 1-step prediction)
         * @param dt time elapsed in seconds (defaults to 0.01s / 10ms)
         * @return float control output
         */
        float update(float error, float velocity, float accel = 0, float dt = 0.01f);

        /**
         * @brief Update the LQR controller with error and dt (estimates velocity internally if external vel is not provided)
         *
         * @param error target minus current position/angle (e)
         * @param dt time elapsed in seconds
         * @return float control output
         */
        float update(float error, float dt = 0.01f);

        /**
         * @brief Reset controller state (integral, filtered velocity, prevError)
         */
        void reset();

        /**
         * @brief Set controller gains dynamically
         */
        void setGains(float kP, float kV, float kA, float kI = 0);

        float getKP() const { return kP; }
        float getKV() const { return kV; }
        float getKA() const { return kA; }
        float getKI() const { return kI; }

    protected:
        float kP;
        float kV;
        float kA;
        float kI;
        float windupRange;
        bool signFlipReset;

        float integral = 0;
        float prevError = 0;
        float filteredVelocity = 0;
        bool isFirstStep = true;
};

} // namespace lemlib
