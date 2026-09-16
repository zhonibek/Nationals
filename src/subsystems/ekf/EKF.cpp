#include "subsystems/ekf/EKF.hpp"
#include "lemlib/util.hpp"
#include <cmath>

namespace lemlib {

constexpr double METER_TO_INCH = 39.37007874;
constexpr double INCH_TO_METER = 0.0254;

double RobotEKF::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

RobotEKF::RobotEKF(const Pose& initialPose) {
    reset(initialPose);
}

void RobotEKF::reset(const Pose& resetPose) {
    ekfMutex.take(TIMEOUT_MAX);

    x_est.setZero();
    x_est(0) = resetPose.x * INCH_TO_METER;
    x_est(1) = resetPose.y * INCH_TO_METER;
    x_est(2) = degToRad(resetPose.theta);
    x_est(3) = 0.0;
    x_est(4) = 0.0;

    P_cov = Eigen::Matrix<double, 5, 5>::Identity();
    P_cov(0, 0) = 0.01;
    P_cov(1, 1) = 0.01;
    P_cov(2, 2) = 0.005;
    P_cov(3, 3) = 0.1;
    P_cov(4, 4) = 0.1;

    Q_noise = Eigen::Matrix<double, 5, 5>::Zero();
    Q_noise(0, 0) = 0.0001; // pos x noise
    Q_noise(1, 1) = 0.0001; // pos y noise
    Q_noise(2, 2) = 0.00005; // heading noise
    Q_noise(3, 3) = 0.01;   // linear vel noise
    Q_noise(4, 4) = 0.01;   // angular vel noise

    ekfMutex.give();
}

void RobotEKF::predict(double dt) {
    if (dt <= 1e-4) dt = 0.01;
    ekfMutex.take(TIMEOUT_MAX);

    double x = x_est(0);
    double y = x_est(1);
    double theta = x_est(2);
    double v = x_est(3);
    double w = x_est(4);

    // 1. Non-linear state propagation (Unicycle kinematics)
    x_est(0) = x + v * std::cos(theta) * dt;
    x_est(1) = y + v * std::sin(theta) * dt;
    x_est(2) = normalizeAngle(theta + w * dt);
    // v and w are assumed constant velocity model + process noise

    // 2. Jacobian of state transition function F
    Eigen::Matrix<double, 5, 5> F = Eigen::Matrix<double, 5, 5>::Identity();
    F(0, 2) = -v * std::sin(theta) * dt;
    F(0, 3) = std::cos(theta) * dt;
    F(1, 2) = v * std::cos(theta) * dt;
    F(1, 3) = std::sin(theta) * dt;
    F(2, 4) = dt;

    // 3. Covariance propagation: P = F * P * F^T + Q
    P_cov = F * P_cov * F.transpose() + Q_noise;
    P_cov = 0.5 * (P_cov + P_cov.transpose()); // guarantee symmetry

    ekfMutex.give();
}

void RobotEKF::updatePose(double odomXMeters, double odomYMeters, double odomThetaRad,
                          double stdDevPos, double stdDevTheta) {
    ekfMutex.take(TIMEOUT_MAX);

    // Measurement matrix H: maps [x, y, theta, v, w] -> [x, y, theta]
    Eigen::Matrix<double, 3, 5> H = Eigen::Matrix<double, 3, 5>::Zero();
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    H(2, 2) = 1.0;

    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    R(0, 0) = stdDevPos * stdDevPos;
    R(1, 1) = stdDevPos * stdDevPos;
    R(2, 2) = stdDevTheta * stdDevTheta;

    // Innovation (measurement residual)
    Eigen::Vector3d z(odomXMeters, odomYMeters, odomThetaRad);
    Eigen::Vector3d y = z - H * x_est;
    y(2) = normalizeAngle(y(2)); // angle residual wrap

    // Innovation covariance: S = H * P * H^T + R
    Eigen::Matrix3d S = H * P_cov * H.transpose() + R;

    // Kalman Gain: K = P * H^T * S^-1
    Eigen::Matrix<double, 5, 3> K = P_cov * H.transpose() * S.inverse();

    // State update: x = x + K * y
    x_est = x_est + K * y;
    x_est(2) = normalizeAngle(x_est(2));

    // Covariance update: P = (I - K * H) * P
    Eigen::Matrix<double, 5, 5> I = Eigen::Matrix<double, 5, 5>::Identity();
    P_cov = (I - K * H) * P_cov;
    P_cov = 0.5 * (P_cov + P_cov.transpose());

    ekfMutex.give();
}

void RobotEKF::updateIMU(double gyroRateRadPerSec, double stdDevGyro) {
    ekfMutex.take(TIMEOUT_MAX);

    // Measurement matrix H: maps [x, y, theta, v, w] -> [w]
    Eigen::Matrix<double, 1, 5> H = Eigen::Matrix<double, 1, 5>::Zero();
    H(0, 4) = 1.0;

    double R = stdDevGyro * stdDevGyro;
    double y = gyroRateRadPerSec - x_est(4);

    double S = (H * P_cov * H.transpose())(0, 0) + R;
    Eigen::Matrix<double, 5, 1> K = P_cov * H.transpose() / S;

    x_est = x_est + K * y;
    x_est(2) = normalizeAngle(x_est(2));

    Eigen::Matrix<double, 5, 5> I = Eigen::Matrix<double, 5, 5>::Identity();
    P_cov = (I - K * H) * P_cov;
    P_cov = 0.5 * (P_cov + P_cov.transpose());

    ekfMutex.give();
}

void RobotEKF::updateWheelVelocities(double leftMps, double rightMps, double trackWidthMeters,
                                     double stdDevVel) {
    ekfMutex.take(TIMEOUT_MAX);

    double forwardVel = (leftMps + rightMps) / 2.0;
    double turnRate = (rightMps - leftMps) / (trackWidthMeters > 1e-4 ? trackWidthMeters : 0.3);

    // Measurement matrix H: maps [x, y, theta, v, w] -> [v, w]
    Eigen::Matrix<double, 2, 5> H = Eigen::Matrix<double, 2, 5>::Zero();
    H(0, 3) = 1.0;
    H(1, 4) = 1.0;

    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    R(0, 0) = stdDevVel * stdDevVel;
    R(1, 1) = (stdDevVel * stdDevVel) / (trackWidthMeters * trackWidthMeters);

    Eigen::Vector2d z(forwardVel, turnRate);
    Eigen::Vector2d y = z - H * x_est;

    Eigen::Matrix2d S = H * P_cov * H.transpose() + R;
    Eigen::Matrix<double, 5, 2> K = P_cov * H.transpose() * S.inverse();

    x_est = x_est + K * y;
    x_est(2) = normalizeAngle(x_est(2));

    Eigen::Matrix<double, 5, 5> I = Eigen::Matrix<double, 5, 5>::Identity();
    P_cov = (I - K * H) * P_cov;
    P_cov = 0.5 * (P_cov + P_cov.transpose());

    ekfMutex.give();
}

Pose RobotEKF::getPose() const {
    ekfMutex.take(TIMEOUT_MAX);
    double x_in = x_est(0) * METER_TO_INCH;
    double y_in = x_est(1) * METER_TO_INCH;
    double theta_deg = radToDeg(x_est(2));
    ekfMutex.give();
    return Pose(x_in, y_in, theta_deg);
}

double RobotEKF::getLinearVelocity() const {
    ekfMutex.take(TIMEOUT_MAX);
    double v = x_est(3);
    ekfMutex.give();
    return v;
}

double RobotEKF::getAngularVelocity() const {
    ekfMutex.take(TIMEOUT_MAX);
    double w = x_est(4);
    ekfMutex.give();
    return w;
}

} // namespace lemlib
