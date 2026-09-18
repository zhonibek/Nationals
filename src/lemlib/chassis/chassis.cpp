#include <math.h>
#include "pros/imu.hpp"
#include "pros/motors.h"
#include "pros/rtos.h"
#include "lemlib/logger/logger.hpp"
#include "lemlib/util.hpp"
#include "lemlib/chassis/chassis.hpp"
#include "lemlib/chassis/odom.hpp"
#include "lemlib/chassis/trackingWheel.hpp"
#include "pros/rtos.hpp"

lemlib::OdomSensors::OdomSensors(TrackingWheel* vertical1, TrackingWheel* vertical2, TrackingWheel* horizontal1,
                                 TrackingWheel* horizontal2, pros::Imu* imu)
    : vertical1(vertical1),
      vertical2(vertical2),
      horizontal1(horizontal1),
      horizontal2(horizontal2),
      imu(imu) {}

lemlib::Drivetrain::Drivetrain(pros::MotorGroup* leftMotors, pros::MotorGroup* rightMotors, float trackWidth,
                               float wheelDiameter, float rpm, float horizontalDrift)
    : leftMotors(leftMotors),
      rightMotors(rightMotors),
      trackWidth(trackWidth),
      wheelDiameter(wheelDiameter),
      rpm(rpm),
      horizontalDrift(horizontalDrift) {}

lemlib::Chassis::Chassis(Drivetrain drivetrain, ControllerSettings linearSettings, ControllerSettings angularSettings,
                         OdomSensors sensors, DriveCurve* throttleCurve, DriveCurve* steerCurve)
    : drivetrain(drivetrain),
      lateralSettings(linearSettings),
      angularSettings(angularSettings),
      lateralLQRSettings(linearSettings.kP, linearSettings.kD, 0, linearSettings.kI, linearSettings.windupRange,
                         linearSettings.smallError, linearSettings.smallErrorTimeout, linearSettings.largeError,
                         linearSettings.largeErrorTimeout, linearSettings.slew, true),
      angularLQRSettings(angularSettings.kP, angularSettings.kD, 0, angularSettings.kI, angularSettings.windupRange,
                         angularSettings.smallError, angularSettings.smallErrorTimeout, angularSettings.largeError,
                         angularSettings.largeErrorTimeout, angularSettings.slew, true),
      sensors(sensors),
      throttleCurve(throttleCurve),
      steerCurve(steerCurve),
      lateralPID(linearSettings.kP, linearSettings.kI, linearSettings.kD, linearSettings.windupRange, true),
      angularPID(angularSettings.kP, angularSettings.kI, angularSettings.kD, angularSettings.windupRange, true),
      lateralLQR(linearSettings.kP, linearSettings.kD, 0, linearSettings.kI, linearSettings.windupRange, true),
      angularLQR(angularSettings.kP, angularSettings.kD, 0, angularSettings.kI, angularSettings.windupRange, true),
      lateralLargeExit(lateralSettings.largeError, lateralSettings.largeErrorTimeout),
      lateralSmallExit(lateralSettings.smallError, lateralSettings.smallErrorTimeout),
      angularLargeExit(angularSettings.largeError, angularSettings.largeErrorTimeout),
      angularSmallExit(angularSettings.smallError, angularSettings.smallErrorTimeout) {}

lemlib::Chassis::Chassis(Drivetrain drivetrain, ControllerSettings linearSettings, ControllerSettings angularSettings,
                         LQRSettings lateralLQRSettings, LQRSettings angularLQRSettings,
                         OdomSensors sensors, DriveCurve* throttleCurve, DriveCurve* steerCurve)
    : drivetrain(drivetrain),
      lateralSettings(linearSettings),
      angularSettings(angularSettings),
      lateralLQRSettings(lateralLQRSettings),
      angularLQRSettings(angularLQRSettings),
      sensors(sensors),
      throttleCurve(throttleCurve),
      steerCurve(steerCurve),
      lateralPID(linearSettings.kP, linearSettings.kI, linearSettings.kD, linearSettings.windupRange, true),
      angularPID(angularSettings.kP, angularSettings.kI, angularSettings.kD, angularSettings.windupRange, true),
      lateralLQR(lateralLQRSettings.kP, lateralLQRSettings.kV, lateralLQRSettings.kA, lateralLQRSettings.kI,
                 lateralLQRSettings.windupRange, true),
      angularLQR(angularLQRSettings.kP, angularLQRSettings.kV, angularLQRSettings.kA, angularLQRSettings.kI,
                 angularLQRSettings.windupRange, true),
      lateralLargeExit(lateralSettings.largeError, lateralSettings.largeErrorTimeout),
      lateralSmallExit(lateralSettings.smallError, lateralSettings.smallErrorTimeout),
      angularLargeExit(angularSettings.largeError, angularSettings.largeErrorTimeout),
      angularSmallExit(angularSettings.smallError, angularSettings.smallErrorTimeout) {}

/**
 * @brief calibrate the IMU given a sensors struct
 *
 * @param sensors reference to the sensors struct
 */
void calibrateIMU(lemlib::OdomSensors& sensors) {
    int attempt = 1;

    // calibrate inertial, and if calibration fails, then repeat 5 times or until successful
    while (attempt <= 5) {
        sensors.imu->reset();
        const auto calibrationStart=pros::millis();
        // wait until IMU is calibrated
        do pros::delay(10);
        while (sensors.imu->get_status() != pros::ImuStatus::error && sensors.imu->is_calibrating() && pros::millis()-calibrationStart<4000);
        // exit if imu has been calibrated
        if (!sensors.imu->is_calibrating() && std::isfinite(sensors.imu->get_heading())) {

            break;
        }
        // indicate error
        pros::c::controller_rumble(pros::E_CONTROLLER_MASTER, "---");
        lemlib::infoSink()->warn("IMU failed to calibrate! Attempt #{}", attempt);
        attempt++;
    }
    // check if calibration attempts were successful
    if (attempt > 5) {
        sensors.imu = nullptr;
        lemlib::infoSink()->error("IMU calibration failed, defaulting to tracking wheels / motor encoders");
    }
}

void lemlib::Chassis::calibrate(bool calibrateImu) {
    // calibrate the IMU if it exists and the user doesn't specify otherwise
    if (sensors.imu != nullptr && calibrateImu) calibrateIMU(sensors);
    // initialize odom
    if (sensors.vertical1 == nullptr)
        sensors.vertical1 = new lemlib::TrackingWheel(drivetrain.leftMotors, drivetrain.wheelDiameter,
                                                      -(drivetrain.trackWidth / 2), drivetrain.rpm);
    if (sensors.vertical2 == nullptr)
        sensors.vertical2 = new lemlib::TrackingWheel(drivetrain.rightMotors, drivetrain.wheelDiameter,
                                                      drivetrain.trackWidth / 2, drivetrain.rpm);
    sensors.vertical1->reset();
    sensors.vertical2->reset();
    if (sensors.horizontal1 != nullptr) sensors.horizontal1->reset();
    if (sensors.horizontal2 != nullptr) sensors.horizontal2->reset();
    setSensors(sensors, drivetrain);
    lemlib::setDrivebaseType(this->drivebaseType);
    init();
    // rumble to controller to indicate success
    pros::c::controller_rumble(pros::E_CONTROLLER_MASTER, ".");
}

void lemlib::Chassis::setDrivebaseType(DrivebaseType type) {
    this->drivebaseType = type;
    lemlib::setDrivebaseType(type);
}

lemlib::DrivebaseType lemlib::Chassis::getDrivebaseType() const {
    return this->drivebaseType;
}

void lemlib::Chassis::setPose(float x, float y, float theta, bool radians) {
    lemlib::setPose(lemlib::Pose(x, y, theta), radians);
}

void lemlib::Chassis::setPose(Pose pose, bool radians) { lemlib::setPose(pose, radians); }

lemlib::Pose lemlib::Chassis::getPose(bool radians, bool standardPos) {
    Pose pose = lemlib::getPose(true);
    if (standardPos) pose.theta = M_PI_2 - pose.theta;
    if (!radians) pose.theta = radToDeg(pose.theta);
    return pose;
}

lemlib::Chassis::~Chassis() {
    cancelAllMotions(); workerRunning=false;
    if(worker) { worker->join(); delete worker; }
}
void lemlib::Chassis::submitMotion(std::function<void()> fn,bool async) {
    uint32_t id;
    {
        Lock guard(mutex);
        if(commands.size()>=16) { result=MotionResult::Busy; return; }
        id=++submitted;
        commands.push_back({id,generation,pros::competition::get_status(),std::move(fn)});
        if(!worker) worker=new pros::Task([this]{runMotions();});
    }
    if(!async) while(completed.load()<id) pros::delay(5);
}
void lemlib::Chassis::runMotions() {
    while(workerRunning) {
        Command cmd{}; bool execute=false;
        {
            Lock guard(mutex);
            if(!commands.empty()) {
                cmd=std::move(commands.front()); commands.pop_front();
                execute=cmd.generation==generation && cmd.mode==pros::competition::get_status();
                active=cmd.id; motionRunning=execute; distTraveled=0;
            }
        }
        if(!cmd.id) { pros::delay(5); continue; }
        result=execute ? MotionResult::Running : MotionResult::Cancelled;
        if(execute && motionRunning) {
            driveOutput().configure(drivetrain.leftMotors,drivetrain.rightMotors);
            driveToken=driveOutput().acquire();
            if(!driveToken) result=getOdomStatus().valid ? MotionResult::Busy : MotionResult::SensorFault;
            else {
                try { cmd.fn(); } catch(...) { result=MotionResult::InvalidInput; }
                driveOutput().release(driveToken); driveToken=0;
            }
        }
        if(result==MotionResult::Running) result=motionRunning ? MotionResult::Settled : MotionResult::Cancelled;
        motionRunning=false; distTraveled=-1; active=0; completed=cmd.id;
    }
}
bool lemlib::Chassis::motionAllowed() {
    if(!motionRunning) return false;
    if(!getOdomStatus().valid) { result=MotionResult::SensorFault; motionRunning=false; return false; }
    if(!driveOutput().owns(driveToken)) { result=MotionResult::Cancelled; motionRunning=false; return false; }
    return true;
}
void lemlib::Chassis::writeDrive(float left,float right) {
    if(!motionRunning || !driveOutput().tank(driveToken,left,right)) {
        motionRunning=false;
        if(result==MotionResult::Running) result=MotionResult::SensorFault;
    }
}
void lemlib::Chassis::endMotion() {} // retained internally for source compatibility
void lemlib::Chassis::waitUntil(float dist) {
    if(!std::isfinite(dist) || dist<0) return;
    const auto id=submitted.load();
    while(completed.load()<id && (active.load()==0 || distTraveled.load()<dist)) pros::delay(5);
}
void lemlib::Chassis::waitUntilDone() {
    const auto id=submitted.load();
    while(completed.load()<id) pros::delay(5);
}
void lemlib::Chassis::cancelMotion() {
    uint32_t id;
    { Lock guard(mutex); id=active.load(); motionRunning=false; }
    // The worker acknowledges completion before another writer can take over.
    while(id && completed.load()<id) pros::delay(5);
}
void lemlib::Chassis::cancelAllMotions() {
    uint32_t id;
    { Lock guard(mutex); ++generation; id=submitted.load(); motionRunning=false; }
    while(completed.load()<id) pros::delay(5);
}
bool lemlib::Chassis::isInMotion() const { return completed.load()<submitted.load(); }

void lemlib::Chassis::resetLocalPosition() {
    float theta = this->getPose().theta;
    lemlib::setPose(lemlib::Pose(0, 0, theta), false);
}

void lemlib::Chassis::setBrakeMode(pros::motor_brake_mode_e mode) {
    drivetrain.leftMotors->set_brake_mode_all(mode);
    drivetrain.rightMotors->set_brake_mode_all(mode);
}

void lemlib::Chassis::setMotionControllerType(MotionControllerType type) {
    this->motionControllerType = type;
}

lemlib::MotionControllerType lemlib::Chassis::getMotionControllerType() const {
    return this->motionControllerType;
}

void lemlib::Chassis::useHybrid() {
    this->motionControllerType = MotionControllerType::HYBRID;
}

void lemlib::Chassis::useLQR(bool enable) {
    this->motionControllerType = enable ? MotionControllerType::LQR : MotionControllerType::PID;
}

void lemlib::Chassis::usePID() {
    this->motionControllerType = MotionControllerType::PID;
}

void lemlib::Chassis::setLateralLQR(LQRSettings settings) {
    this->lateralLQRSettings = settings;
    this->lateralLQR.setGains(settings.kP, settings.kV, settings.kA, settings.kI);
}

void lemlib::Chassis::setAngularLQR(LQRSettings settings) {
    this->angularLQRSettings = settings;
    this->angularLQR.setGains(settings.kP, settings.kV, settings.kA, settings.kI);
}

lemlib::LQR& lemlib::Chassis::getLateralLQR() {
    return this->lateralLQR;
}

lemlib::LQR& lemlib::Chassis::getAngularLQR() {
    return this->angularLQR;
}

