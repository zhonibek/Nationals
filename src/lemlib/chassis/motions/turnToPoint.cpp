#include <cmath>
#include "lemlib/chassis/chassis.hpp"
#include "lemlib/logger/logger.hpp"
#include "lemlib/timer.hpp"
#include "lemlib/util.hpp"
#include "lemlib/chassis/odom.hpp"
#include "pros/misc.hpp"

void lemlib::Chassis::turnToPointImpl(float x, float y, int timeout, TurnToPointParams params, bool async) {
    params.minSpeed = std::abs(params.minSpeed);

    if (getPose().distance(Pose(x,y))<1e-4f) { result=MotionResult::Settled; return; }
    float targetTheta;
    float deltaX, deltaY, deltaTheta;
    float motorPower;
    float prevMotorPower = 0;
    float startTheta = getPose().theta;
    bool settling = false;
    std::optional<float> prevRawDeltaTheta = std::nullopt;
    std::optional<float> prevDeltaTheta = std::nullopt;
    std::uint8_t compState = pros::competition::get_status();
    distTraveled = 0;
    Timer timer(timeout);
    angularLargeExit.reset();
    angularSmallExit.reset();
    angularPID.reset();
    angularLQR.reset();

    LoopClock loopClock;
    // main loop
    while (!timer.isDone() && !angularLargeExit.getExit() && !angularSmallExit.getExit() && motionAllowed()) {
        const float dt=loopClock.tick();
        if(dt>0.1f) { result=MotionResult::SensorFault; break; }
        // update variables
        Pose pose = getPose();
        pose.theta = (params.forwards) ? fmod(pose.theta, 360) : fmod(pose.theta - 180, 360);

        // update completion vars
        distTraveled = fabs(angleError(pose.theta, startTheta, false));

        deltaX = x - pose.x;
        deltaY = y - pose.y;
        targetTheta = fmod(radToDeg(M_PI_2 - atan2(deltaY, deltaX)), 360);

        // check if settling
        const float rawDeltaTheta = angleError(targetTheta, pose.theta, false);
        if (prevRawDeltaTheta == std::nullopt) prevRawDeltaTheta = rawDeltaTheta;
        if (sgn(rawDeltaTheta) != sgn(prevRawDeltaTheta)) settling = true;
        prevRawDeltaTheta = rawDeltaTheta;

        // calculate deltaTheta
        if (settling) deltaTheta = angleError(targetTheta, pose.theta, false);
        else deltaTheta = angleError(targetTheta, pose.theta, false, params.direction);
        if (prevDeltaTheta == std::nullopt) prevDeltaTheta = deltaTheta;

        // motion chaining
        if (params.minSpeed != 0 && fabs(deltaTheta) < params.earlyExitRange) break;
        if (params.minSpeed != 0 && sgn(deltaTheta) != sgn(prevDeltaTheta)) break;

        // calculate the speed
        if (motionControllerType == MotionControllerType::HYBRID) {
            float angularVel = (sensors.imu != nullptr) ? -sensors.imu->get_gyro_rate().z : getLocalSpeed().theta;
            float pidOut = angularPID.update(deltaTheta, dt);
            float lqrDamping = -angularLQRSettings.kV * angularVel;

            // Stiction feedforward: overcomes Coulomb friction to eliminate stall
            float stiction = 0.0f;
            if (std::fabs(deltaTheta) > 0.15f) {
                float kS_turn = 9.0f;
                stiction = std::clamp(deltaTheta / 1.2f, -1.0f, 1.0f) * kS_turn;
            }
            motorPower = pidOut + lqrDamping + stiction;
        } else if (motionControllerType == MotionControllerType::LQR) {
            float angularVel = 0;
            if (sensors.imu != nullptr) {
                angularVel = -sensors.imu->get_gyro_rate().z;
            } else {
                angularVel = getLocalSpeed().theta;
            }
            motorPower = angularLQR.update(deltaTheta, angularVel, 0, dt);
        } else {
            motorPower = angularPID.update(deltaTheta, dt);
        }
        angularLargeExit.update(deltaTheta);
        angularSmallExit.update(deltaTheta);

        // cap the speed
        if (motorPower > params.maxSpeed) motorPower = params.maxSpeed;
        else if (motorPower < -params.maxSpeed) motorPower = -params.maxSpeed;
        if (angularSettings.slew > 0) motorPower = slew(motorPower, prevMotorPower, angularSettings.slew * dt / 0.01f);
        if (motorPower < 0 && motorPower > -params.minSpeed) motorPower = -params.minSpeed;
        else if (motorPower > 0 && motorPower < params.minSpeed) motorPower = params.minSpeed;
        prevMotorPower = motorPower;

        infoSink()->debug("Turn Motor Power: {} ", motorPower);

        // move the drivetrain
        writeDrive(motorPower, -motorPower);

        pros::delay(10);
    }

    if(result==MotionResult::Running) result = !motionRunning ? MotionResult::Cancelled : (timer.isDone() ? MotionResult::TimedOut : MotionResult::Settled);
    // stop the drivetrain
    writeDrive(0, 0);
    // set distTraveled to -1 to indicate that the function has finished
    distTraveled = -1;

}
void lemlib::Chassis::turnToPoint(float x, float y, int timeout, TurnToPointParams params, bool async) {
    if (!std::isfinite(x) || !std::isfinite(y) || timeout<=0 || !std::isfinite(params.maxSpeed) || !std::isfinite(params.minSpeed) || !std::isfinite(params.earlyExitRange) || params.maxSpeed<=0) { result=MotionResult::InvalidInput; return; }
    params.maxSpeed=std::min<float>(params.maxSpeed,127.f);
    params.minSpeed=std::clamp<float>(std::fabs(params.minSpeed),0.f,params.maxSpeed);
    submitMotion([this,x,y,timeout,params] { turnToPointImpl(x,y,timeout,params, false); },async);
}
