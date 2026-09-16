#include "subsystems/trajectory/QuinticSpline.hpp"
#include "lemlib/util.hpp"
#include <cmath>
#include <algorithm>

namespace lemlib {

std::vector<State> QuinticSplineGenerator::generateTrajectory(const SplineWaypoints& params, double dt) {
    if (dt <= 1e-4) dt = 0.02;

    // Convert start and end poses from inches to meters
    double x0 = params.start.x * INCH_TO_METER;
    double y0 = params.start.y * INCH_TO_METER;
    double theta0 = degToRad(params.start.theta);

    double x1 = params.end.x * INCH_TO_METER;
    double y1 = params.end.y * INCH_TO_METER;
    double theta1 = degToRad(params.end.theta);

    double dist = std::hypot(x1 - x0, y1 - y0);
    if (dist < 1e-3) {
        return { State{x0, y0, theta0, 0.0, 0.0} };
    }

    // Tangent vectors scaled by distance to create a pronounced, smooth curvature arc
    double scale = std::max(1.35 * dist, 0.45);
    double vx0 = scale * std::sin(theta0); // In LemLib heading, dx = sin(theta), dy = cos(theta)
    double vy0 = scale * std::cos(theta0);
    double vx1 = scale * std::sin(theta1);
    double vy1 = scale * std::cos(theta1);

    // Initial and final accelerations (0 for smooth start/stop)
    double ax0 = 0.0, ay0 = 0.0;
    double ax1 = 0.0, ay1 = 0.0;

    // Quintic polynomial coefficients for X(s) where s in [0, 1]
    double cx0 = x0;
    double cx1 = vx0;
    double cx2 = 0.5 * ax0;
    double cx3 = 10.0 * (x1 - x0) - (6.0 * vx0 + 4.0 * vx1) - (1.5 * ax0 - 0.5 * ax1);
    double cx4 = -15.0 * (x1 - x0) + (8.0 * vx0 + 7.0 * vx1) + (1.5 * ax0 - ax1);
    double cx5 = 6.0 * (x1 - x0) - 3.0 * (vx0 + vx1) - 0.5 * (ax0 - ax1);

    // Quintic polynomial coefficients for Y(s) where s in [0, 1]
    double cy0 = y0;
    double cy1 = vy0;
    double cy2 = 0.5 * ay0;
    double cy3 = 10.0 * (y1 - y0) - (6.0 * vy0 + 4.0 * vy1) - (1.5 * ay0 - 0.5 * ay1);
    double cy4 = -15.0 * (y1 - y0) + (8.0 * vy0 + 7.0 * vy1) + (1.5 * ay0 - ay1);
    double cy5 = 6.0 * (y1 - y0) - 3.0 * (vy0 + vy1) - 0.5 * (ay0 - ay1);

    // Estimate total time based on trapezoidal / S-curve profile
    double effectiveMaxVel = std::max(params.maxVel, 0.2);
    double totalTime = (1.5 * dist / effectiveMaxVel) + 0.6; // account for acceleration ramp
    int numSteps = std::max(static_cast<int>(std::ceil(totalTime / dt)), 10);

    std::vector<State> trajectory;
    trajectory.reserve(numSteps + 1);

    // Standard Cartesian math angle frame (matches LTV DARE solver effective_theta)
    double prevHeading = M_PI_2 - theta0;

    for (int i = 0; i <= numSteps; ++i) {
        double t = static_cast<double>(i) / numSteps; // normalized time [0, 1]
        double t2 = t * t;
        double t3 = t2 * t;
        double t4 = t3 * t;
        double t5 = t4 * t;

        // Position on curve
        double px = cx0 + cx1 * t + cx2 * t2 + cx3 * t3 + cx4 * t4 + cx5 * t5;
        double py = cy0 + cy1 * t + cy2 * t2 + cy3 * t3 + cy4 * t4 + cy5 * t5;

        // Velocity components w.r.t normalized parameter t
        double dpx = cx1 + 2.0 * cx2 * t + 3.0 * cx3 * t2 + 4.0 * cx4 * t3 + 5.0 * cx5 * t4;
        double dpy = cy1 + 2.0 * cy2 * t + 3.0 * cy3 * t2 + 4.0 * cy4 * t3 + 5.0 * cy5 * t4;

        // Instantaneous Cartesian heading: atan2(dy, dx) matches LTV DARE solver frame
        double heading = std::atan2(dpy, dpx);
        if (std::isnan(heading)) heading = prevHeading;

        // Smooth Jerk-limited S-curve velocity profile
        double s_vel = std::sin(M_PI * t);
        double v = params.maxVel * s_vel;

        // Angular velocity calculation
        double w = 0.0;
        if (i > 0) {
            double dTheta = heading - prevHeading;
            while (dTheta > M_PI) dTheta -= 2.0 * M_PI;
            while (dTheta < -M_PI) dTheta += 2.0 * M_PI;
            w = dTheta / dt;
        }
        prevHeading = heading;

        trajectory.push_back({ px, py, heading, v, w });
    }

    return trajectory;
}

std::vector<State> QuinticSplineGenerator::generateMultiPointTrajectory(
    const std::vector<Pose>& waypoints, double maxVel, double maxAccel, double dt) {
    if (waypoints.size() < 2) return {};

    std::vector<State> fullTrajectory;

    for (size_t i = 0; i < waypoints.size() - 1; ++i) {
        SplineWaypoints segment;
        segment.start = waypoints[i];
        segment.end = waypoints[i + 1];
        segment.maxVel = maxVel;
        segment.maxAccel = maxAccel;

        auto segTrajectory = generateTrajectory(segment, dt);
        if (i > 0 && !segTrajectory.empty()) {
            segTrajectory.erase(segTrajectory.begin()); // prevent duplicate connection points
        }
        fullTrajectory.insert(fullTrajectory.end(), segTrajectory.begin(), segTrajectory.end());
    }

    return fullTrajectory;
}

} // namespace lemlib
