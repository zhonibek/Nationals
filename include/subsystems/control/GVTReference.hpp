#pragma once
#include "Cascade.hpp"
#include "subsystems/pedro/BezierCurve.hpp"
#include "subsystems/pedro/PedroPath.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace nationals {

/**
 * @brief GVT (Guiding Vector Tracking) Reference Generator for LTV-LQR + PID Cascade.
 * 
 * Unlike rigid time-parameterized trajectory generators, GVT operates as a continuous
 * spatial vector field:
 * 1. Fast spatial projection: Finds parameter t* on the Bézier curve closest to the robot.
 * 2. Predictive braking: Computes allowable forward velocity based on true remaining arc length.
 * 3. Centripetal compensation: Applies feedforward acceleration along the curve normal.
 * 4. Decoupled heading: Supports independent heading modes (tangent, linear, constant, facing point).
 * 5. Time-independence: Physical delays or defense bumps do not cause clock drift;
 *    the reference state dynamically pulls the robot back via LTV-LQR Riccati state feedback.
 */
class GVTReference {
public:
    pedro::PedroPath path;
    double maxSpeed = 0.45;        // Nominal maximum velocity (m/s)
    double maxAccel = 0.80;        // Maximum linear acceleration (m/s^2)
    double maxJerk = 3.50;         // Maximum allowable jerk (m/s^3)
    double centripetalGain = 0.08; // Feedforward gain for centripetal force
    float currentT = 0.0f;
    size_t currentSegment = 0;
    double totalLengthMeters = 0.0;
    double estimatedDuration = 1.0;

    GVTReference(const pedro::PedroPath& path, double maxSpeed = 0.45, double maxAccel = 0.80, double maxJerk = 3.50)
        : path(path), maxSpeed(maxSpeed), maxAccel(maxAccel), maxJerk(maxJerk) {
        init();
    }

    GVTReference(const pedro::BezierCurve& curve, float startHeadingDeg, float endHeadingDeg,
                 double maxSpeed = 0.45, double maxAccel = 0.80, double maxJerk = 3.50)
        : maxSpeed(maxSpeed), maxAccel(maxAccel), maxJerk(maxJerk) {
        path.addSegment(curve);
        path.setLinearHeading(startHeadingDeg, endHeadingDeg);
        init();
    }

    void reset() {
        currentT = 0.0f;
        currentSegment = 0;
    }

    /**
     * @brief Computes spatial GVT reference state based on current robot feedback.
     * 
     * @param fb Current fused robot odometry and wheel feedback
     * @param isFinished Output flag indicating if robot has reached the end of the path
     * @return Reference Target pose, linear velocity vector, and heading rate for LTV-LQR Cascade
     */
    Reference update(const Feedback& fb, bool& isFinished) {
        isFinished = false;
        if (!fb.valid || path.getSegments().empty()) {
            isFinished = true;
            return {{NAN, NAN, NAN}, 0, 0, 0};
        }

        const auto& segments = path.getSegments();
        if (currentSegment >= segments.size()) {
            isFinished = true;
            const auto& lastCurve = segments.back();
            auto endPt = lastCurve.getPoint(1.0f);
            float finalH = path.getTargetHeading(segments.size() - 1, 1.0f, endPt);
            return {{endPt.x * 0.0254, endPt.y * 0.0254, finalH * pi / 180.0}, 0, 0, 0};
        }

        const auto& curve = segments[currentSegment];

        // 1. Convert robot position from meters to inches for Bézier projection
        pedro::Point robotPos(fb.pose.x / 0.0254, fb.pose.y / 0.0254);

        // 2. Fast spatial projection onto current Bézier segment
        currentT = curve.project(robotPos, currentT);
        pedro::Point pathPoint = curve.getPoint(currentT);
        pedro::Vector2D tangent = curve.getTangent(currentT);
        pedro::Vector2D normal = curve.getNormal(currentT);
        float curvature = curve.getCurvature(currentT); // 1/inches

        // 3. Multi-curve segment transition logic
        if (currentT >= 0.995f && currentSegment < segments.size() - 1) {
            float distToEnd = robotPos.distanceTo(curve.getPoint(1.0f));
            if (distToEnd <= 2.5f) { // Within 2.5 inches of segment boundary
                currentSegment++;
                currentT = 0.0f;
                return update(fb, isFinished);
            }
        }

        // 4. Compute true remaining distance along trajectory (in meters)
        float remInches = curve.getRemainingDistance(currentT);
        for (size_t s = currentSegment + 1; s < segments.size(); ++s) {
            remInches += segments[s].getLength();
        }
        double remMeters = remInches * 0.0254;

        // 5. Pedro predictive braking speed profile
        double vBrake = std::sqrt(2.0 * maxAccel * std::max(0.0, remMeters));
        double curvatureMeters = std::abs(curvature) / 0.0254;
        double vCentripetal = (curvatureMeters > 1e-4) ? std::sqrt(maxAccel / curvatureMeters) : maxSpeed;
        double targetSpeed = std::min({maxSpeed, vBrake, vCentripetal});

        // Minimum crawl speed until final settle tolerance
        if (remMeters > 0.025 && targetSpeed < 0.06) {
            targetSpeed = 0.06;
        }

        // 6. GVT Tangential Velocity + Centripetal Compensation Vector
        double vx = tangent.x * targetSpeed + normal.x * (targetSpeed * targetSpeed * (curvature / 0.0254) * centripetalGain);
        double vy = tangent.y * targetSpeed + normal.y * (targetSpeed * targetSpeed * (curvature / 0.0254) * centripetalGain);

        // 7. Decoupled Heading from PedroPath
        float targetHeadingDeg = path.getTargetHeading(currentSegment, currentT, robotPos);
        double targetHeadingRad = targetHeadingDeg * pi / 180.0;

        // Angular feedforward rate
        double w_ff = 0.0;
        if (totalLengthMeters > 1e-3) {
            double headingDelta = wrap(targetHeadingRad - fb.pose.h);
            w_ff = (headingDelta / totalLengthMeters) * targetSpeed;
            w_ff = std::clamp(w_ff, -2.5, 2.5);
        }

        // 8. Termination evaluation on final segment
        if (currentSegment == segments.size() - 1 && currentT >= 0.99f && remMeters <= 0.025) {
            isFinished = true;
        }

        return {{pathPoint.x * 0.0254, pathPoint.y * 0.0254, targetHeadingRad}, vx, vy, w_ff};
    }

private:
    void init() {
        totalLengthMeters = 0.0;
        for (const auto& seg : path.getSegments()) {
            totalLengthMeters += seg.getLength() * 0.0254;
        }
        if (totalLengthMeters > 1e-6 && maxSpeed > 1e-4) {
            estimatedDuration = (1.875 * totalLengthMeters / maxSpeed) + 0.5;
        } else {
            estimatedDuration = 1.0;
        }
    }
};

} // namespace nationals
