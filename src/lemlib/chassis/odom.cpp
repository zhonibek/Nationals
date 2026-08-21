// The implementation below is mostly based off of
// the document written by 5225A (Pilons)
// Here is a link to the original document
// http://thepilons.ca/wp-content/uploads/2018/10/Tracking.pdf

#include <math.h>
#include "pros/rtos.hpp"
#include "lemlib/util.hpp"
#include "lemlib/chassis/odom.hpp"
#include "lemlib/chassis/chassis.hpp"
#include "lemlib/chassis/trackingWheel.hpp"

// tracking thread
pros::Task* trackingTask = nullptr;

// global variables
static pros::Mutex odomMutex;
lemlib::OdomSensors odomSensors(nullptr, nullptr, nullptr, nullptr, nullptr); // the sensors to be used for odometry
lemlib::Drivetrain drive(nullptr, nullptr, 0, 0, 0, 0); // the drivetrain to be used for odometry
lemlib::Pose odomPose(0, 0, 0); // the pose of the robot
lemlib::Pose odomSpeed(0, 0, 0); // the speed of the robot
lemlib::Pose odomLocalSpeed(0, 0, 0); // the local speed of the robot

float prevVertical = 0;
float prevVertical1 = 0;
float prevVertical2 = 0;
float prevHorizontal = 0;
float prevHorizontal1 = 0;
float prevHorizontal2 = 0;
float prevImu = 0;

void lemlib::setSensors(lemlib::OdomSensors sensors, lemlib::Drivetrain drivetrain) {
    odomMutex.take(TIMEOUT_MAX);
    odomSensors = sensors;
    drive = drivetrain;
    odomMutex.give();
}

lemlib::Pose lemlib::getPose(bool radians) {
    odomMutex.take(TIMEOUT_MAX);
    lemlib::Pose p = odomPose;
    odomMutex.give();
    if (radians) return p;
    else return lemlib::Pose(p.x, p.y, radToDeg(p.theta));
}

void lemlib::setPose(lemlib::Pose pose, bool radians) {
    odomMutex.take(TIMEOUT_MAX);
    if (radians) odomPose = pose;
    else odomPose = lemlib::Pose(pose.x, pose.y, degToRad(pose.theta));
    odomMutex.give();
}

lemlib::Pose lemlib::getSpeed(bool radians) {
    odomMutex.take(TIMEOUT_MAX);
    lemlib::Pose s = odomSpeed;
    odomMutex.give();
    if (radians) return s;
    else return lemlib::Pose(s.x, s.y, radToDeg(s.theta));
}

lemlib::Pose lemlib::getLocalSpeed(bool radians) {
    odomMutex.take(TIMEOUT_MAX);
    lemlib::Pose ls = odomLocalSpeed;
    odomMutex.give();
    if (radians) return ls;
    else return lemlib::Pose(ls.x, ls.y, radToDeg(ls.theta));
}

lemlib::Pose lemlib::estimatePose(float time, bool radians) {
    // get current position and speed
    Pose curPose = getPose(true);
    Pose localSpeed = getLocalSpeed(true);
    // calculate the change in local position
    Pose deltaLocalPose = localSpeed * time;

    // calculate the future pose
    float avgHeading = curPose.theta + deltaLocalPose.theta / 2;
    Pose futurePose = curPose;
    futurePose.x += deltaLocalPose.y * sin(avgHeading);
    futurePose.y += deltaLocalPose.y * cos(avgHeading);
    futurePose.x += deltaLocalPose.x * -cos(avgHeading);
    futurePose.y += deltaLocalPose.x * sin(avgHeading);
    if (!radians) futurePose.theta = radToDeg(futurePose.theta);

    return futurePose;
}

void lemlib::update() {
    odomMutex.take(TIMEOUT_MAX);
    // get the current sensor values
    float vertical1Raw = 0;
    float vertical2Raw = 0;
    float horizontal1Raw = 0;
    float horizontal2Raw = 0;
    float imuRaw = 0;
    if (odomSensors.vertical1 != nullptr) vertical1Raw = odomSensors.vertical1->getDistanceTraveled();
    if (odomSensors.vertical2 != nullptr) vertical2Raw = odomSensors.vertical2->getDistanceTraveled();
    if (odomSensors.horizontal1 != nullptr) horizontal1Raw = odomSensors.horizontal1->getDistanceTraveled();
    if (odomSensors.horizontal2 != nullptr) horizontal2Raw = odomSensors.horizontal2->getDistanceTraveled();
    if (odomSensors.imu != nullptr) imuRaw = degToRad(odomSensors.imu->get_rotation());

    // calculate the change in sensor values
    float deltaVertical1 = vertical1Raw - prevVertical1;
    float deltaVertical2 = vertical2Raw - prevVertical2;
    float deltaHorizontal1 = horizontal1Raw - prevHorizontal1;
    float deltaHorizontal2 = horizontal2Raw - prevHorizontal2;
    float deltaImu = imuRaw - prevImu;

    // update the previous sensor values
    prevVertical1 = vertical1Raw;
    prevVertical2 = vertical2Raw;
    prevHorizontal1 = horizontal1Raw;
    prevHorizontal2 = horizontal2Raw;
    prevImu = imuRaw;

    // calculate the heading of the robot
    // Priority:
    // 1. Horizontal tracking wheels
    // 2. Vertical tracking wheels (unpowered)
    // 3. Inertial Sensor
    // 4. Drivetrain motor encoders
    float heading = odomPose.theta;
    if (odomSensors.horizontal1 != nullptr && odomSensors.horizontal2 != nullptr) {
        float offsetDiff = odomSensors.horizontal1->getOffset() - odomSensors.horizontal2->getOffset();
        if (std::abs(offsetDiff) > 1e-4f) {
            heading -= (deltaHorizontal1 - deltaHorizontal2) / offsetDiff;
        }
    } else if (odomSensors.vertical1 != nullptr && odomSensors.vertical2 != nullptr &&
               !odomSensors.vertical1->getType() && !odomSensors.vertical2->getType()) {
        float offsetDiff = odomSensors.vertical1->getOffset() - odomSensors.vertical2->getOffset();
        if (std::abs(offsetDiff) > 1e-4f) {
            heading -= (deltaVertical1 - deltaVertical2) / offsetDiff;
        }
    } else if (odomSensors.imu != nullptr) {
        heading += deltaImu;
    } else if (odomSensors.vertical1 != nullptr && odomSensors.vertical2 != nullptr) {
        float offsetDiff = odomSensors.vertical1->getOffset() - odomSensors.vertical2->getOffset();
        if (std::abs(offsetDiff) > 1e-4f) {
            heading -= (deltaVertical1 - deltaVertical2) / offsetDiff;
        }
    }
    float deltaHeading = heading - odomPose.theta;
    float avgHeading = odomPose.theta + deltaHeading / 2.0f;

    // choose tracking wheels to use (prioritize non-powered tracking wheels)
    lemlib::TrackingWheel* verticalWheel = nullptr;
    lemlib::TrackingWheel* horizontalWheel = nullptr;
    if (odomSensors.vertical1 != nullptr && !odomSensors.vertical1->getType()) verticalWheel = odomSensors.vertical1;
    else if (odomSensors.vertical2 != nullptr && !odomSensors.vertical2->getType()) verticalWheel = odomSensors.vertical2;
    else if (odomSensors.vertical1 != nullptr) verticalWheel = odomSensors.vertical1;
    else if (odomSensors.vertical2 != nullptr) verticalWheel = odomSensors.vertical2;

    if (odomSensors.horizontal1 != nullptr) horizontalWheel = odomSensors.horizontal1;
    else if (odomSensors.horizontal2 != nullptr) horizontalWheel = odomSensors.horizontal2;

    float rawVertical = 0;
    float rawHorizontal = 0;
    if (verticalWheel != nullptr) rawVertical = verticalWheel->getDistanceTraveled();
    if (horizontalWheel != nullptr) rawHorizontal = horizontalWheel->getDistanceTraveled();
    float horizontalOffset = 0;
    float verticalOffset = 0;
    if (verticalWheel != nullptr) verticalOffset = verticalWheel->getOffset();
    if (horizontalWheel != nullptr) horizontalOffset = horizontalWheel->getOffset();

    // calculate change in x and y
    float deltaX = 0;
    float deltaY = 0;
    if (verticalWheel != nullptr) deltaY = rawVertical - prevVertical;
    if (horizontalWheel != nullptr) deltaX = rawHorizontal - prevHorizontal;
    prevVertical = rawVertical;
    prevHorizontal = rawHorizontal;

    // calculate local x and y
    float localX = 0;
    float localY = 0;
    if (std::abs(deltaHeading) < 1e-6f) { // prevent divide by 0
        localX = deltaX;
        localY = deltaY;
    } else {
        localX = 2.0f * sin(deltaHeading / 2.0f) * (deltaX / deltaHeading + horizontalOffset);
        localY = 2.0f * sin(deltaHeading / 2.0f) * (deltaY / deltaHeading + verticalOffset);
    }

    // save previous pose
    lemlib::Pose prevPose = odomPose;

    // calculate global x and y
    odomPose.x += localY * sin(avgHeading);
    odomPose.y += localY * cos(avgHeading);
    odomPose.x += localX * -cos(avgHeading);
    odomPose.y += localX * sin(avgHeading);
    odomPose.theta = heading;

    // calculate speed
    odomSpeed.x = ema((odomPose.x - prevPose.x) / 0.01f, odomSpeed.x, 0.95f);
    odomSpeed.y = ema((odomPose.y - prevPose.y) / 0.01f, odomSpeed.y, 0.95f);
    odomSpeed.theta = ema((odomPose.theta - prevPose.theta) / 0.01f, odomSpeed.theta, 0.95f);

    // calculate local speed
    odomLocalSpeed.x = ema(localX / 0.01f, odomLocalSpeed.x, 0.95f);
    odomLocalSpeed.y = ema(localY / 0.01f, odomLocalSpeed.y, 0.95f);
    odomLocalSpeed.theta = ema(deltaHeading / 0.01f, odomLocalSpeed.theta, 0.95f);
    odomMutex.give();
}

void lemlib::init() {
    if (trackingTask == nullptr) {
        trackingTask = new pros::Task {[=] {
            while (true) {
                update();
                pros::delay(10);
            }
        }};
    }
}
