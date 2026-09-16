#pragma once

#include <Eigen/Dense>
#include "pros/rtos.hpp"
#include "lemlib/pose.hpp"

namespace lemlib {

/**
 * @brief 5-State Extended Kalman Filter (EKF) for Robot Localization and Sensor Fusion.
 *
 * State vector: x = [X (m), Y (m), Theta (rad), v (m/s), w (rad/s)]^T
 * Fuses motor encoders, tracking wheels, and IMU gyro/accel.
 */
class RobotEKF {
public:
    /**
     * @brief Construct a new Robot EKF
     * @param initialPose Initial robot pose
     */
    RobotEKF(const Pose& initialPose = Pose(0, 0, 0));

    /**
     * @brief Reset the state and covariance matrices
     */
    void reset(const Pose& resetPose = Pose(0, 0, 0));

    /**
     * @brief Time update (Prediction step using kinematic unicycle motion model)
     * @param dt Time step in seconds
     */
    void predict(double dt = 0.01);

    /**
     * @brief Measurement update from odometry pose
     * @param odomXMeters Odometry X position in meters
     * @param odomYMeters Odometry Y position in meters
     * @param odomThetaRad Odometry Theta in radians
     * @param stdDevPos Position measurement standard deviation in meters
     * @param stdDevTheta Heading measurement standard deviation in radians
     */
    void updatePose(double odomXMeters, double odomYMeters, double odomThetaRad,
                    double stdDevPos = 0.03, double stdDevTheta = 0.02);

    /**
     * @brief Measurement update from IMU Gyroscope rate
     * @param gyroRateRadPerSec Angular velocity around Z axis in radians/sec
     * @param stdDevGyro Gyro measurement standard deviation
     */
    void updateIMU(double gyroRateRadPerSec, double stdDevGyro = 0.01);

    /**
     * @brief Measurement update from differential drive wheel speeds
     * @param leftMps Left wheel speed in meters/second
     * @param rightMps Right wheel speed in meters/second
     * @param trackWidthMeters Distance between wheel tracks in meters
     * @param stdDevVel Velocity measurement standard deviation
     */
    void updateWheelVelocities(double leftMps, double rightMps, double trackWidthMeters,
                               double stdDevVel = 0.05);

    /**
     * @brief Get the fused optimal pose in LemLib coordinates (inches, degrees)
     */
    Pose getPose() const;

    /**
     * @brief Get linear velocity in m/s
     */
    double getLinearVelocity() const;

    /**
     * @brief Get angular velocity in rad/s
     */
    double getAngularVelocity() const;

private:
    // State vector: [x, y, theta, v, w]
    Eigen::Matrix<double, 5, 1> x_est;

    // State covariance matrix
    Eigen::Matrix<double, 5, 5> P_cov;

    // Process noise covariance matrix
    Eigen::Matrix<double, 5, 5> Q_noise;

    mutable pros::Mutex ekfMutex;

    static double normalizeAngle(double angle);
};

} // namespace lemlib
