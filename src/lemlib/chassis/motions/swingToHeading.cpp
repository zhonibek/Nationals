#include <cmath>
#include "lemlib/chassis/chassis.hpp"
#include "lemlib/logger/logger.hpp"
#include "lemlib/timer.hpp"
#include "lemlib/util.hpp"
#include "lemlib/chassis/odom.hpp"
#include "pros/misc.hpp"

void lemlib::Chassis::swingToHeadingImpl(float theta, DriveSide lockedSide, int timeout, SwingToHeadingParams params,
                                     bool async) {
    params.minSpeed = fabs(params.minSpeed);

    float targetTheta;
    float deltaTheta;
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
    // get original braking mode of that side of the drivetrain so we can set it back to it after this motion ends
    pros::MotorBrake brakeMode = (lockedSide == DriveSide::LEFT)
                                     ? this->drivetrain.leftMotors->get_brake_mode_all().at(0)
                                     : this->drivetrain.rightMotors->get_brake_mode_all().at(0);
    // set brake mode of the locked side to hold
    if (lockedSide == DriveSide::LEFT) this->drivetrain.leftMotors->set_brake_mode_all(pros::E_MOTOR_BRAKE_HOLD);
    else this->drivetrain.rightMotors->set_brake_mode_all(pros::E_MOTOR_BRAKE_HOLD);

    LoopClock loopClock;
    // main loop
    while (!timer.isDone() && !angularLargeExit.getExit() && !angularSmallExit.getExit() && motionAllowed()) {
        const float dt=loopClock.tick();
        if(dt>0.1f) { result=MotionResult::SensorFault; break; }
        // update variables
        Pose pose = getPose();
        pose.theta = fmod(pose.theta, 360);

        // update completion vars
        distTraveled = fabs(angleError(pose.theta, startTheta, false));
        targetTheta = theta;

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
        if (motionControllerType == MotionControllerType::LQR) {
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
        if (fabs(deltaTheta) > 20) motorPower = slew(motorPower, prevMotorPower, angularSettings.slew * dt / 0.01f);
        if (motorPower < 0 && motorPower > -params.minSpeed) motorPower = -params.minSpeed;
        else if (motorPower > 0 && motorPower < params.minSpeed) motorPower = params.minSpeed;
        prevMotorPower = motorPower;

        infoSink()->debug("Turn Motor Power: {} ", motorPower);

        // move the drivetrain
        if (lockedSide == DriveSide::LEFT) {
            writeDrive(0, -motorPower);
        } else {
            writeDrive(motorPower, 0);
        }

        // delay to save resources
        pros::delay(10);
    }

    // set the brake mode of the locked side of the drivetrain to its
    // original value
    if (lockedSide == DriveSide::LEFT) this->drivetrain.leftMotors->set_brake_mode_all(brakeMode);
    else this->drivetrain.rightMotors->set_brake_mode_all(brakeMode);
    if(result==MotionResult::Running) result = !motionRunning ? MotionResult::Cancelled : (timer.isDone() ? MotionResult::TimedOut : MotionResult::Settled);
    // stop the drivetrain
    writeDrive(0, 0);
    // set distTraveled to -1 to indicate that the function has finished
    distTraveled = -1;

}
void lemlib::Chassis::swingToHeading(float theta, DriveSide lockedSide, int timeout, SwingToHeadingParams params,
                                     bool async) {
    if (!std::isfinite(theta) || timeout<=0 || !std::isfinite(params.maxSpeed) || !std::isfinite(params.minSpeed) || !std::isfinite(params.earlyExitRange) || params.maxSpeed<=0) { result=MotionResult::InvalidInput; return; }
    params.maxSpeed=std::min<float>(params.maxSpeed,127.f);
    params.minSpeed=std::clamp<float>(std::fabs(params.minSpeed),0.f,params.maxSpeed);
    submitMotion([this,theta,lockedSide,timeout,params] { swingToHeadingImpl(theta,lockedSide,timeout,params, false); },async);
}
