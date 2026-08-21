#include "subsystems/VelocityController.hpp"

VelocityController::VelocityController(const VelocityControllerConfig& config)
    : config(config) {}

VelocityController::VelocityController(double kV, double kA_straight, double kA_turn, double kS_straight, double kS_turn,
                                       double kP_straight, double kI_straight, double max_voltage, double trackWidthMeters) {
    config.kV = kV;
    config.KA_straight = kA_straight;
    config.KA_turn = kA_turn;
    config.KS_straight = kS_straight;
    config.KS_turn = kS_turn;
    config.KP_straight = kP_straight;
    config.KI_straight = kI_straight;
    config.max_voltage = max_voltage;
    config.trackWidthMeters = trackWidthMeters;
}

void VelocityController::reset() {
    prev_v_cmd = 0.0;
    prev_w_cmd = 0.0;
    left_integral = 0.0;
    right_integral = 0.0;
}

void VelocityController::setConfig(const VelocityControllerConfig& newConfig) {
    config = newConfig;
}

DrivetrainVoltages VelocityController::update(double v_cmd, double w_cmd, double left_actual_mps, double right_actual_mps, double dt) {
    if (dt <= 0.0001) dt = 0.01;

    // Convert unicycle velocities (v, w) into differential wheel velocities (v_left, v_right)
    double halfTrack = config.trackWidthMeters / 2.0;
    double v_left_target = v_cmd - (w_cmd * halfTrack);
    double v_right_target = v_cmd + (w_cmd * halfTrack);

    // Compute instantaneous acceleration commands (dv/dt, dw/dt)
    double a_lin = (v_cmd - prev_v_cmd) / dt;
    double a_ang = (w_cmd - prev_w_cmd) / dt;
    prev_v_cmd = v_cmd;
    prev_w_cmd = w_cmd;

    double a_left_target = a_lin - (a_ang * halfTrack);
    double a_right_target = a_lin + (a_ang * halfTrack);

    // Dynamic static friction overcoming (blends between straight and turning kS)
    double turnRatio = (std::abs(v_cmd) + std::abs(w_cmd) > 1e-4) ? std::abs(w_cmd) / (std::abs(v_cmd) + std::abs(w_cmd)) : 0.0;
    double kS_eff = config.KS_straight * (1.0 - turnRatio) + config.KS_turn * turnRatio;
    double kA_eff = config.KA_straight * (1.0 - turnRatio) + config.KA_turn * turnRatio;

    // Smooth feedforward voltages (smooth tanh avoids 100Hz square-wave buzzing at near-zero speeds)
    double ff_left = (kS_eff * std::tanh(v_left_target / 0.05))
                   + (config.kV * v_left_target)
                   + (kA_eff * a_left_target);

    double ff_right = (kS_eff * std::tanh(v_right_target / 0.05))
                    + (config.kV * v_right_target)
                    + (kA_eff * a_right_target);

    // Velocity tracking errors
    double err_left = v_left_target - left_actual_mps;
    double err_right = v_right_target - right_actual_mps;

    // Integrate errors with anti-windup
    left_integral += err_left * dt;
    right_integral += err_right * dt;
    left_integral = clamp(left_integral, -2.0, 2.0);
    right_integral = clamp(right_integral, -2.0, 2.0);

    // Feedback voltages (PI controller)
    double fb_left = (config.KP_straight * err_left) + (config.KI_straight * left_integral);
    double fb_right = (config.KP_straight * err_right) + (config.KI_straight * right_integral);

    double total_left = clamp(ff_left + fb_left, -config.max_voltage, config.max_voltage);
    double total_right = clamp(ff_right + fb_right, -config.max_voltage, config.max_voltage);

    return { total_left, total_right };
}
