#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include "pros/rtos.hpp"
#include "pros/motor_group.hpp"
#include "lemlib/chassis/chassis.hpp"
#include "subsystems/VelocityController.hpp"
#include "subsystems/ltv/State.hpp"
#include "Eigen/Dense"

namespace lemlib {

constexpr double INCH_TO_METER = 0.0254;
constexpr double METER_TO_INCH = 1.0 / 0.0254;

/**
 * @brief Configuration parameters for the Linear Time-Varying (LTV) LQR path follower
 */
struct ltvConfig {
    // Forward state error penalty weights (Q diagonal)
    float q_x = 1.5f;
    float q_y = 6.0f;
    float q_theta = 3.5f;

    // Forward control effort penalty weights (R diagonal)
    float r_vel = 0.008f;
    float r_ang = 0.006f;

    // Backward state & control weights
    float q_x_b = 1.5f;
    float q_y_b = 6.0f;
    float q_theta_b = 3.5f;
    float r_vel_b = 0.008f;
    float r_ang_b = 0.006f;

    float q_scalar = 1.0f;
    float max_lin_correction = 1.5f; // m/s limit on feedback correction
    float max_ang_correction = 4.0f; // rad/s limit on feedback correction

    bool backwards = false;
    bool turnFirst = false;
    bool test = false;
    bool log = true;
};

/**
 * @brief 2D Vector utility for LaTeX / telemetry logging
 */
struct Vector2 {
    float x;
    float y;
    Vector2(float x, float y) : x(x), y(y) {}
    std::string latex() const;
};

/**
 * @brief Linear Time-Varying (LTV) Path Follower utilizing an online discrete algebraic Riccati equation (DARE) solver.
 */
class LTVPathFollower {
public:
    LTVPathFollower(Chassis& chassis, pros::MotorGroup& leftMotors, pros::MotorGroup& rightMotors,
                    const VelocityControllerConfig& config);

    /**
     * @brief Asynchronously follow a precomputed or parsed trajectory
     */
    void followPath(const std::string& path_name, const ltvConfig& l_config = {});

    /**
     * @brief Asynchronously follow a raw State vector trajectory
     */
    void followTrajectory(const std::vector<State>& trajectory, const ltvConfig& l_config = {});

    /**
     * @brief Precompute and parse path trajectories in the background to save runtime CPU cycles
     */
    void precompute_paths(const std::vector<std::string>& path_names);

    /**
     * @brief Wait until path execution completes
     */
    void waitUntilDone();

    /**
     * @brief Wait until robot travels a certain distance in inches
     */
    void waitUntil(float dist_inches);

    /**
     * @brief Wait until robot is within radius of a target waypoint
     */
    void waitUntil(float x_inch, float y_inch, float radius_inch);

    /**
     * @brief Abort current path execution
     */
    void cancel();

    /**
     * @brief Returns whether LTV path following is currently active
     */
    bool isRunning();

    /**
     * @brief Calculate path length in inches
     */
    double getPathLength(const std::string& path_name);

    static std::vector<State> prepare_trajectory(const std::string& data);

private:
    Chassis& chassis;
    pros::MotorGroup& leftMotors;
    pros::MotorGroup& rightMotors;
    VelocityController controller;

    float rpm_to_mps_factor;
    bool is_running = false;
    bool cancel_request = false;
    bool abortAuton = false;
    float distance_traveled_inches = 0.0f;

    std::unordered_map<std::string, std::vector<State>> precomputed_paths;
    pros::Task* task = nullptr;

    struct TaskParams {
        LTVPathFollower* instance;
        std::string path_name;
        ltvConfig config;
        std::vector<State> dynamic_path;
    };

    static void task_trampoline(void* params);
    static void precompute_paths_task(void* param);

    void followPathImpl(const std::string& path_name, const ltvConfig& l_config, const std::vector<State>& dynamic_path);

    static double angleError(double robotAngle, double targetAngle);
    static double clamp(double value, double min, double max);

    static Eigen::MatrixXf dareSolver(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B, const Eigen::MatrixXf &Q, const Eigen::MatrixXf &R);
    static std::pair<Eigen::MatrixXf, Eigen::MatrixXf> discretizeAB(const Eigen::MatrixXf& contA, const Eigen::MatrixXf& contB, double dtSeconds);
    static std::vector<std::vector<double>> parse_tuples(const std::string& line);
};

} // namespace lemlib
