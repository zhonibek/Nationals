#include "subsystems/pedro/PedroFollower.hpp"
#include "pros/rtos.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include "pros/misc.hpp"

namespace pedro {

PedroFollower::PedroFollower(FollowerConfig config) : config(config) {}

lemlib::MotionResult PedroFollower::follow(const BezierCurve& curve,
                           HeadingMode headingMode,
                           float targetHeadingDeg,
                           int timeoutMs,
                           std::function<void(int forward, int strafe, int turn)> driveFn,
                           std::function<void()> brakeFn) {
    PedroPath path(curve, headingMode, targetHeadingDeg);
    return follow(path, timeoutMs, driveFn, brakeFn);
}

lemlib::MotionResult PedroFollower::follow(const PedroPath& path,
                           int timeoutMs,
                           std::function<void(int forward, int strafe, int turn)> driveFn,
                           std::function<void()> brakeFn) {
    if(running.exchange(true)) return lemlib::MotionResult::Busy;
    struct Finish { std::atomic<bool>& running; std::function<void()>& brake; ~Finish(){ if(brake) brake(); running=false; } } finish{running,brakeFn};
    const FollowerConfig config=getConfig();
    const auto& segments = path.getSegments();
    if(segments.empty() || timeoutMs<=0 || !driveFn || !brakeFn ||
       !std::isfinite(config.maxVel)||config.maxVel<=0||config.maxVel>127 ||
       !std::isfinite(config.maxAccel)||config.maxAccel<=0 || config.minVel<0||config.minVel>config.maxVel ||
       config.exitDistance<=0 || config.exitHeadingError<=0 || config.settleTimeoutMs<0)
        return lemlib::MotionResult::InvalidInput;
    for(const auto& segment:segments) if(!std::isfinite(segment.getLength())) return lemlib::MotionResult::InvalidInput;
    const auto mode=pros::competition::get_status();
    lemlib::LoopClock loopClock;
    bool firstHeading=true;
    float filteredDerivative=0;

    cancelled = false;
    uint32_t startTime = pros::millis();
    size_t currentSegment = 0;
    float currentT = 0.0f;

    // Heading PID states
    float headingIntegral = 0.0f;
    float prevHeadingError = 0.0f;
    uint32_t settleStartTime = 0;
    bool isSettling = false;

    // Loop at 10ms (100 Hz)


    while (pros::millis() - startTime < static_cast<uint32_t>(timeoutMs) && !cancelled) {
        const float dt=loopClock.tick();
        if(pros::competition::get_status()!=mode || pros::competition::is_disabled()) return lemlib::MotionResult::Cancelled;
        const auto snapshot=lemlib::getOdomSnapshot();
        if(!snapshot.status.valid || dt>0.1f) return lemlib::MotionResult::SensorFault;
        lemlib::Pose curPose = snapshot.pose; curPose.theta=lemlib::radToDeg(curPose.theta);
        Point robotPos(curPose.x, curPose.y);
        float robotHeadingDeg = curPose.theta;
        float robotHeadingRad = lemlib::degToRad(robotHeadingDeg);

        const BezierCurve& curve = segments[currentSegment];

        // 1. Fast projection onto current curve segment
        currentT = curve.project(robotPos, currentT);
        Point pathPoint = curve.getPoint(currentT);
        Vector2D tangent = curve.getTangent(currentT);
        Vector2D normal = curve.getNormal(currentT);
        float curvature = curve.getCurvature(currentT);

        // 2. Segment transition logic for multi-curve paths
        if (currentT >= 0.999f && robotPos.distanceTo(curve.getPoint(1))<=config.exitDistance && currentSegment < segments.size() - 1) {
            currentSegment++;
            currentT = 0.0f;
            continue;
        }

        // 3. Compute remaining distance to end of trajectory
        float remainingDist = curve.getRemainingDistance(currentT);
        for (size_t s = currentSegment + 1; s < segments.size(); ++s) {
            remainingDist += segments[s].getLength();
        }

        // 4. Predictive Braking Speed Profile: v = min(v_max, sqrt(2 * a * s))
        float brakingSpeed = std::sqrt(2.0f * config.maxAccel * std::max(0.0f, remainingDist));
        float targetSpeed = std::min(config.maxVel, brakingSpeed);

        if (remainingDist > config.exitDistance && targetSpeed < config.minVel) {
            targetSpeed = config.minVel;
        }

        // 5. Drive Vector along path tangent
        Vector2D driveVector = tangent * targetSpeed;

        // 6. Translational Corrective Vector (Guiding Vector Field / GVF)
        Vector2D errorVector = pathPoint - robotPos;
        Vector2D correctionVector = errorVector * config.kP_trans;

        // 7. Centripetal Force Feedforward: a_c = v^2 * kappa * normal
        Vector2D centripetalVector = normal * (targetSpeed * targetSpeed * curvature * config.k_centripetal);

        // 8. Combined Field Translational Demand Vector
        Vector2D fieldDemand = driveVector + correctionVector + centripetalVector;

        // 9. Transform Field Demand to Robot Local Coordinates (Forward & Strafe)
        // LemLib frame: 0 deg = North (+Y), 90 deg = East (+X)
        float localForward = fieldDemand.x * std::sin(robotHeadingRad) + fieldDemand.y * std::cos(robotHeadingRad);
        float localStrafe  = fieldDemand.x * std::cos(robotHeadingRad) - fieldDemand.y * std::sin(robotHeadingRad);

        // LQR Optimal Velocity State Damping (cancels physical momentum and prevents jitter)
        lemlib::Pose localVel = snapshot.localSpeed;
        localForward -= config.kD_lqr * localVel.y;
        localStrafe  -= config.kD_lqr * localVel.x;

        // Static friction feedforward (kS)
        if (std::abs(localForward) > 0.5f) {
            localForward += (localForward > 0 ? config.kS : -config.kS);
        }
        if (std::abs(localStrafe) > 0.5f) {
            localStrafe += (localStrafe > 0 ? config.kS : -config.kS);
        }

        // Clamp combined translational demand
        float transMag = std::hypot(localForward, localStrafe);
        if (transMag > config.maxVel) {
            localForward = (localForward / transMag) * config.maxVel;
            localStrafe  = (localStrafe / transMag) * config.maxVel;
        }

        // 10. Decoupled Heading Controller (PID + LQR Angular Rate Damping)
        float targetHeadingDeg = path.getTargetHeading(currentSegment, currentT, robotPos);
        float headingError = std::remainder(targetHeadingDeg - robotHeadingDeg, 360.0f);

        if (std::abs(headingError) < 15.0f) {
            headingIntegral += headingError * dt;
            headingIntegral = std::clamp(headingIntegral, -20.0f, 20.0f);
        } else {
            headingIntegral = 0.0f;
        }

        const float rawDerivative=firstHeading?0:std::remainder(headingError-prevHeadingError,360.f)/dt;
        firstHeading=false;
        filteredDerivative+=(1-std::exp(-dt/0.03f))*(rawDerivative-filteredDerivative);
        float headingDerivative=filteredDerivative;
        prevHeadingError = headingError;

        float turnCmd = config.headingKp * headingError +
                        config.headingKi * headingIntegral +
                        config.headingKd * headingDerivative -
                        0.18f * lemlib::radToDeg(localVel.theta); // LQR yaw rate damping
        turnCmd = std::clamp(turnCmd, -90.0f, 90.0f);

        // 11. Exit & Settle Evaluation (only on final segment)
        bool onFinalSegment = (currentSegment == segments.size() - 1);
        float posError = errorVector.magnitude();

        if (onFinalSegment && remainingDist <= config.exitDistance &&
            posError <= config.exitDistance && std::abs(headingError) <= config.exitHeadingError &&
            std::hypot(localVel.x,localVel.y)<=config.exitVelocity && std::abs(lemlib::radToDeg(localVel.theta))<=config.exitAngularVelocity) {
            if (!isSettling) {
                isSettling = true;
                settleStartTime = pros::millis();
            } else if (pros::millis() - settleStartTime >= static_cast<uint32_t>(config.settleTimeoutMs)) {
                return lemlib::MotionResult::Settled;
            }
        } else {
            isSettling = false;
        }

        // 12. Send motor command via callback
        if(!std::isfinite(localForward)||!std::isfinite(localStrafe)||!std::isfinite(turnCmd)) return lemlib::MotionResult::InvalidInput;
        const float wheelPeak=std::max({std::abs(localForward+localStrafe+turnCmd),std::abs(localForward-localStrafe+turnCmd),std::abs(localForward-localStrafe-turnCmd),std::abs(localForward+localStrafe-turnCmd)});
        const float scale=std::max(1.f,wheelPeak/config.maxVel);
        localForward/=scale; localStrafe/=scale; turnCmd/=scale;
        driveFn(static_cast<int>(localForward), static_cast<int>(localStrafe), static_cast<int>(turnCmd));

        pros::delay(10);
    }

    // Motion complete: apply brakes
    return cancelled ? lemlib::MotionResult::Cancelled : lemlib::MotionResult::TimedOut;
}

} // namespace pedro
