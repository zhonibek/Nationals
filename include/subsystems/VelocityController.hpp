#pragma once

#include <cmath>

struct DrivetrainVoltages {
    double leftVoltage;  // in Volts (e.g. -12.0 to +12.0)
    double rightVoltage; // in Volts (e.g. -12.0 to +12.0)
};

struct VelocityControllerConfig {
    double kV = 4.5;           // Velocity feedforward gain (V / (m/s))
    double KA_straight = 0.2;  // Acceleration feedforward gain straight (V / (m/s^2))
    double KA_turn = 0.15;     // Acceleration feedforward gain turning
    double KS_straight = 0.4;  // Static friction overcome voltage straight (V)
    double KS_turn = 0.3;      // Static friction overcome voltage turning (V)
    double KP_straight = 2.0;  // Velocity error proportional gain
    double KI_straight = 0.0;  // Velocity error integral gain
    double max_voltage = 12.0; // Maximum voltage limit (V)
    double trackWidthMeters = 0.2667; // 10.5 inches in meters
    bool enableTCS = true;     // Active Traction Control System (Anti-Slip)
    double maxSlipRatio = 0.18; // Maximum allowable slip ratio before torque intervention
};

/**
 * @brief Inner-loop PIDf voltage controller.
 * Converts linear and angular velocity targets (v_cmd in m/s, w_cmd in rad/s)
 * into left and right motor voltages (V) utilizing feedforward (kS, kV, kA) + feedback PID.
 */
class VelocityController {
public:
    VelocityController(const VelocityControllerConfig& config);
    VelocityController(double kV, double kA_straight, double kA_turn, double kS_straight, double kS_turn,
                       double kP_straight, double kI_straight, double max_voltage, double trackWidthMeters);

    /**
     * @brief Update the velocity controller to compute required motor voltages.
     * 
     * @param v_cmd Target linear velocity (m/s)
     * @param w_cmd Target angular velocity (rad/s)
     * @param left_actual_mps Current measured left wheel velocity (m/s)
     * @param right_actual_mps Current measured right wheel velocity (m/s)
     * @param dt Time delta in seconds (defaults to 0.01s)
     * @return DrivetrainVoltages Voltages for left and right motors in Volts
     */
    DrivetrainVoltages update(double v_cmd, double w_cmd, double left_actual_mps, double right_actual_mps, double dt = 0.01);

    void reset();
    void setConfig(const VelocityControllerConfig& newConfig);

private:
    VelocityControllerConfig config;
    double prev_v_cmd = 0.0;
    double prev_w_cmd = 0.0;
    double left_integral = 0.0;
    double right_integral = 0.0;

    static double sgn(double val) {
        if (val > 0.0) return 1.0;
        if (val < 0.0) return -1.0;
        return 0.0;
    }

    static double clamp(double val, double min, double max) {
        if (val < min) return min;
        if (val > max) return max;
        return val;
    }
};
