#include <cmath>
#include "lemlib/chassis/chassis.hpp"
#include "lemlib/logger/logger.hpp"
#include "lemlib/timer.hpp"
#include "lemlib/util.hpp"
#include "lemlib/chassis/odom.hpp"
#include "pros/misc.hpp"

void lemlib::Chassis::moveToPointImpl(float x, float y, int timeout, MoveToPointParams params, bool async) {
    params.earlyExitRange = fabs(params.earlyExitRange);


    // reset controllers and exit conditions
    lateralPID.reset();
    lateralLargeExit.reset();
    lateralSmallExit.reset();
    angularPID.reset();
    lateralLQR.reset();
    angularLQR.reset();

    // initialize vars used between iterations
    Pose lastPose = getPose();
    distTraveled = 0;
    Timer timer(timeout);
    bool close = false;
    float prevLateralOut = 0; // previous lateral power
    float prevAngularOut = 0; // previous angular power
    const int compState = pros::competition::get_status();
    std::optional<bool> prevSide = std::nullopt;

    // calculate target pose in standard form
    Pose target(x, y);
    target.theta = lastPose.angle(target);

    LoopClock loopClock;
    // main loop
    while (!timer.isDone() && ((!lateralSmallExit.getExit() && !lateralLargeExit.getExit()) || !close) &&
           motionAllowed()) {
        const float dt=loopClock.tick();
        if(dt>0.1f) { result=MotionResult::SensorFault; break; }
        // update position
        const Pose pose = getPose(true, true);

        // update distance traveled
        distTraveled += pose.distance(lastPose);
        lastPose = pose;

        // calculate distance to the target point
        const float distTarget = pose.distance(target);

        // check if the robot is close enough to the target to start settling
        if (distTarget < 7.5 && close == false) {
            close = true;
            // Keep the caller's power cap during settling.
        }

        // motion chaining
        const bool side =
            (pose.y - target.y) * -sin(target.theta) <= (pose.x - target.x) * cos(target.theta) + params.earlyExitRange;
        if (prevSide == std::nullopt) prevSide = side;
        const bool sameSide = side == prevSide;
        // exit if close
        if (!sameSide && params.minSpeed != 0) break;
        prevSide = side;

        // calculate error
        const float adjustedRobotTheta = params.forwards ? pose.theta : pose.theta + M_PI;
        const float angularError = angleError(adjustedRobotTheta, pose.angle(target));
        float lateralError = pose.distance(target) * cos(angleError(pose.theta, pose.angle(target)));

        // update exit conditions
        lateralSmallExit.update(distTarget);
        lateralLargeExit.update(distTarget);

        // get output from active controller (HYBRID, LQR, or PID)
        float lateralOut = 0;
        float angularOut = 0;
        if (motionControllerType == MotionControllerType::HYBRID) {
            float forwardVel = getLocalSpeed(true).y;
            float lqrLateralDamping = -lateralLQRSettings.kV * forwardVel;

            // Stiction Feedforward: Coulomb static friction compensation for linear drive
            // When error is within stall zone (~1-2"), smoothly injects ~12.0 units of power
            // to eliminate the 1.5" friction stall, fading to 0 within 0.15" deadband.
            float stictionLat = 0.0f;
            if (std::fabs(lateralError) > 0.15f) {
                float kS_drive = 12.0f;
                stictionLat = std::clamp(lateralError / 1.0f, -1.0f, 1.0f) * kS_drive;
            }
            lateralOut = lateralPID.update(lateralError, dt) + lqrLateralDamping + stictionLat;

            float angularVel = (sensors.imu != nullptr) ? -sensors.imu->get_gyro_rate().z : getLocalSpeed().theta;
            float lqrAngularDamping = -angularLQRSettings.kV * angularVel;
            float angErrDeg = radToDeg(angularError);
            float stictionAng = 0.0f;
            if (std::fabs(angErrDeg) > 0.15f) {
                float kS_turn = 8.5f;
                stictionAng = std::clamp(angErrDeg / 1.2f, -1.0f, 1.0f) * kS_turn;
            }
            angularOut = angularPID.update(angErrDeg, dt) + lqrAngularDamping + stictionAng;
        } else if (motionControllerType == MotionControllerType::LQR) {
            float forwardVel = getLocalSpeed(true).y;
            float forwardAccel = 0;
            if (sensors.imu != nullptr && lateralLQRSettings.useImuPrediction) {
                pros::imu_accel_s_t accel = sensors.imu->get_accel();
                forwardAccel = accel.y * 386.08858f; // Gs to in/s^2 for 1-step prediction
            }
            lateralOut = lateralLQR.update(lateralError, forwardVel, forwardAccel, dt);

            float angularVel = 0;
            if (sensors.imu != nullptr) {
                angularVel = -sensors.imu->get_gyro_rate().z;
            } else {
                angularVel = getLocalSpeed().theta;
            }
            angularOut = angularLQR.update(radToDeg(angularError), angularVel, 0, dt);
        } else {
            lateralOut = lateralPID.update(lateralError, dt);
            angularOut = angularPID.update(radToDeg(angularError), dt);
        }
        if (distTarget < lateralSettings.smallError) angularOut = 0;

        // apply restrictions on angular speed
        angularOut = std::clamp(angularOut, -params.maxSpeed, params.maxSpeed);
        angularOut = slew(angularOut, prevAngularOut, angularSettings.slew * dt / 0.01f);

        // apply restrictions on lateral speed
        lateralOut = std::clamp(lateralOut, -params.maxSpeed, params.maxSpeed);
        // constrain lateral output by max accel
        // but not for decelerating, since that would interfere with settling
        if (!close) lateralOut = slew(lateralOut, prevLateralOut, lateralSettings.slew * dt / 0.01f);

        // prevent moving in the wrong direction
        if (params.forwards && !close) lateralOut = std::fmax(lateralOut, 0);
        else if (!params.forwards && !close) lateralOut = std::fmin(lateralOut, 0);

        // constrain lateral output by the minimum speed
        if (params.forwards && lateralOut < fabs(params.minSpeed) && lateralOut > 0) lateralOut = fabs(params.minSpeed);
        if (!params.forwards && -lateralOut < fabs(params.minSpeed) && lateralOut < 0)
            lateralOut = -fabs(params.minSpeed);

        // update previous output
        prevAngularOut = angularOut;
        prevLateralOut = lateralOut;

        infoSink()->debug("Angular Out: {}, Lateral Out: {}", angularOut, lateralOut);

        // ratio the speeds to respect the max speed
        float leftPower = lateralOut + angularOut;
        float rightPower = lateralOut - angularOut;
        const float ratio = std::max(std::fabs(leftPower), std::fabs(rightPower)) / params.maxSpeed;
        if (ratio > 1) {
            leftPower /= ratio;
            rightPower /= ratio;
        }

        // move the drivetrain
        writeDrive(leftPower, rightPower);

        // delay to save resources
        pros::delay(10);
    }

    if(result==MotionResult::Running) result = !motionRunning ? MotionResult::Cancelled : (timer.isDone() ? MotionResult::TimedOut : MotionResult::Settled);
    // stop the drivetrain
    writeDrive(0, 0);
    // set distTraveled to -1 to indicate that the function has finished
    distTraveled = -1;

}
void lemlib::Chassis::moveToPoint(float x, float y, int timeout, MoveToPointParams params, bool async) {
    if (!std::isfinite(x) || !std::isfinite(y) || timeout<=0 || !std::isfinite(params.maxSpeed) || !std::isfinite(params.minSpeed) || !std::isfinite(params.earlyExitRange) || params.maxSpeed<=0) { result=MotionResult::InvalidInput; return; }
    params.maxSpeed=std::min<float>(params.maxSpeed,127.f);
    params.minSpeed=std::clamp<float>(std::fabs(params.minSpeed),0.f,params.maxSpeed);
    submitMotion([this,x,y,timeout,params] { moveToPointImpl(x,y,timeout,params, false); },async);
}
