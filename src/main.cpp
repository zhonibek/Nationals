#include "main.h"
#include "lemlib/api.hpp"
#include "lemlib/chassis/odom.hpp"
#include "subsystems/MotorMonitor.hpp"
#include "subsystems/subsystems.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>

// ============================================================================
//  VEX V5 X-DRIVE TUNING & CALIBRATION SUITE
// ============================================================================
//  Hardware Configuration:
//    - Base: Holonomic X-Drive (45-degree omni wheels)
//    - Cartridges: 200 RPM (Green)
//    - Front Left:  Port -11 (reversed)
//    - Back Left:   Port -20 (reversed)
//    - Front Right: Port 1   (normal)
//    - Back Right:  Port 10  (normal)
//
//  Controller Quick Button Reference:
//    [UP]    -> Test Motor Directions (FL, BL, FR, BR one-by-one + holistic)
//    [A]     -> Track Width Calibration (360-degree spin test)
//    [B]     -> Linear Distance Drive Test (24 inches forward)
//    [Y]     -> Angular Heading Turn Test (90 degrees snap turn)
//    [X]     -> Lateral Strafe Test (24 inches sideways)
//    [DOWN]  -> Feedforward Friction Characterization (kS measurement)
//    [LEFT]  -> 180-Degree Snap Turn Test
//    [RIGHT] -> Reset Odometry Pose to (0, 0, 0)
//
//  Controller: Unified HYBRID LQR + PID (Active by default, no toggle needed!)
//
//  Joysticks (Driver Control):
//    - Left Stick Y:  Forward is Backward, Backward is Forward (Inverted)
//    - Left Stick X:  Strafe Left / Right (Push Right = Move Right)
//    - Right Stick X: Yaw Turn Clockwise / Counter-Clockwise (Reversed)
// ============================================================================

// ============================================================================
// 1. Hardware & Motors
// ============================================================================
pros::Controller controller(pros::E_CONTROLLER_MASTER);

// Individual motors for 4-wheel X-Drive
pros::Motor frontLeft(-11, pros::MotorGearset::green);
pros::Motor backLeft(-20, pros::MotorGearset::green);
pros::Motor frontRight(1, pros::MotorGearset::green);
pros::Motor backRight(10, pros::MotorGearset::green);

// Paired motor groups for LemLib differential drive representation
pros::MotorGroup leftMotors({-11, -20}, pros::MotorGearset::green);
pros::MotorGroup rightMotors({1, 10}, pros::MotorGearset::green);

// Watchdog monitoring motor temperatures, disconnections, and overcurrent
lemlib::MotorMonitor motorMonitor(controller, {{"LeftDrive", &leftMotors}, {"RightDrive", &rightMotors}});

// ============================================================================
// 2. Drivetrain Geometry & Constants (X-Drive Kinematics)
// ============================================================================
constexpr float SQRT_2 = 1.41421356f;

// Physical dimensions (measured on your robot):
constexpr float PHYSICAL_WHEEL_DIAMETER = 3.25f; // 3.25" omni wheels (change to 4.0f or 2.75f if used)
constexpr float PHYSICAL_TRACK_WIDTH    = 10.5f; // Distance from center of left wheels to center of right wheels
constexpr float DRIVETRAIN_RPM          = 200.0f;

// Effective dimensions for LemLib:
// 1. Effective Wheel Diameter = Physical Diameter * sqrt(2) (linear distance 1:1)
constexpr float EFFECTIVE_WHEEL_DIAMETER = PHYSICAL_WHEEL_DIAMETER * SQRT_2; // ≈ 4.596" for 3.25" wheels
// 2. Effective Track Width for X-Drive turning in LemLib = 2 * Physical Width (21.0")
//    Because LemLib calculates angle from differential wheels, on an X-Drive the displacement
//    is doubled. Setting 2 * PHYSICAL_TRACK_WIDTH fixes the 45° vs 90° under-turn!
constexpr float TRACK_WIDTH_INCHES       = PHYSICAL_TRACK_WIDTH * 2.0f;     // 21.0"

// Optional VEX Inertial Sensor (IMU)
// HIGHLY RECOMMENDED for X-Drive! If an IMU is plugged into port 21, uncomment:
// pros::Imu imu(21);

lemlib::Drivetrain drivetrain(&leftMotors,
                              &rightMotors,
                              TRACK_WIDTH_INCHES,
                              EFFECTIVE_WHEEL_DIAMETER,
                              DRIVETRAIN_RPM,
                              2.0f);

// ============================================================================
// 3. Motion Controllers (Unified HYBRID: LQR + PID)
// ============================================================================
// In HYBRID mode:
//   - PID handles setpoint guidance (kP) and eliminates steady-state offset (kI).
//   - LQR handles optimal dynamic velocity damping (kV) to cancel momentum without jitter.
//   - Continuous Slew Rate Limiting (15.0f) eliminates motor throttling/chattering!

// Lateral PID Controller (Forward/Backward distance)
lemlib::ControllerSettings linearController(9.5f,  // kP: responsive setpoint tracking
                                            0.02f, // kI: steady-state offset eliminator
                                            2.5f,  // kD: smooth deceleration
                                            2.0f,  // anti-windup range (inches)
                                            0.35f, // small error range (inches) - high precision settle band
                                            100,   // small error timeout (ms)
                                            0.9f,  // large error range (inches) - tightened from 2.5" to prevent early exit
                                            250,   // large error timeout (ms)
                                            15.0f  // slew rate limit: smooth voltage ramping
);

// Angular PID Controller (Heading / Turning)
lemlib::ControllerSettings angularController(3.4f,  // kP: crisp turn response
                                             0.05f, // kI: gently removes remaining offset
                                             2.0f,  // kD: dampens oscillation
                                             2.0f,  // anti-windup range (degrees)
                                             0.4f,  // small error range (degrees) - high precision settle band
                                             100,   // small error timeout (ms)
                                             1.0f,  // large error range (degrees) - tightened from 2.5° to prevent early exit
                                             250,   // large error timeout (ms)
                                             15.0f  // slew rate limit: stops violent motor direction slamming
);

// Lateral LQR Settings (Optimal State Velocity Damping)
lemlib::LQRSettings lateralLQR(9.5f,  // kP: matching proportional gain
                               0.42f, // kV: smooth velocity damping
                               0.0f,  // kA
                               0.02f, // kI
                               2.0f,  // windup range
                               0.35f, 100, 0.9f, 250, 15.0f, false);

// Angular LQR Settings (Optimal Yaw Rate Damping)
lemlib::LQRSettings angularLQR(3.4f,  // kP: matching turn gain
                               0.18f, // kV: optimal yaw rate damping (cancels angular momentum)
                               0.0f,  // kA
                               0.05f, // kI
                               2.0f,  // windup range
                               0.4f, 100, 1.0f, 250, 15.0f, false);

// Odometry Sensors (motor encoder tracking by default)
// If you have a VEX Inertial Sensor (IMU), uncomment the imu definition above and use:
// lemlib::OdomSensors sensors(nullptr, nullptr, nullptr, nullptr, &imu);
lemlib::OdomSensors sensors(nullptr, nullptr, nullptr, nullptr, nullptr);

// Exponential Drive Curves for smooth driver feel
lemlib::ExpoDriveCurve throttleCurve(3, 10, 1.019);
lemlib::ExpoDriveCurve steerCurve(3, 10, 1.019);

// Chassis Instance
lemlib::Chassis chassis(drivetrain, linearController, angularController, lateralLQR, angularLQR, sensors,
                        &throttleCurve, &steerCurve);

// ============================================================================
// 4. Holonomic X-Drive Kinematics
// ============================================================================

/**
 * @brief 3-DOF Holonomic Drive by power [-127, 127]
 * @param throttle Forward (+) / Backward (-)
 * @param strafe   Right (+) / Left (-)
 * @param turn     Clockwise (+) / Counter-Clockwise (-)
 */
void holonomicDrive(int throttle, int strafe, int turn) {
    // Correct X-Drive wheel kinematics: positive strafe = right
    int s = strafe;
    int fl = throttle + s + turn;
    int bl = throttle - s + turn;
    int fr = throttle - s - turn;
    int br = throttle + s - turn;

    int maxMag = std::max({std::abs(fl), std::abs(bl), std::abs(fr), std::abs(br), 127});
    if (maxMag > 127) {
        fl = (fl * 127) / maxMag;
        bl = (bl * 127) / maxMag;
        fr = (fr * 127) / maxMag;
        br = (br * 127) / maxMag;
    }

    frontLeft.move(fl);
    backLeft.move(bl);
    frontRight.move(fr);
    backRight.move(br);
}

/**
 * @brief Stop all 4 motors and apply active brake mode
 */
void holonomicBrake(pros::motor_brake_mode_e mode = pros::E_MOTOR_BRAKE_BRAKE) {
    frontLeft.set_brake_mode(mode);
    backLeft.set_brake_mode(mode);
    frontRight.set_brake_mode(mode);
    backRight.set_brake_mode(mode);
    frontLeft.brake();
    backLeft.brake();
    frontRight.brake();
    backRight.brake();
}

/**
 * @brief Autonomous lateral strafe test along a locked heading
 */
void holonomicStrafe(float strafeInches, float headingDeg = 0.0f, int timeout = 2500, float maxSpeed = 100.0f) {
    uint32_t startTime = pros::millis();
    lemlib::Pose startPose = chassis.getPose();
    float headingRad = lemlib::degToRad(headingDeg);

    float targetX = startPose.x + strafeInches * std::cos(headingRad);
    float targetY = startPose.y - strafeInches * std::sin(headingRad);

    while (pros::millis() - startTime < static_cast<uint32_t>(timeout)) {
        lemlib::Pose cur = chassis.getPose();
        float dx = targetX - cur.x;
        float dy = targetY - cur.y;
        float distErr = std::hypot(dx, dy);
        if (distErr < 0.8f) break;

        float curRad = lemlib::degToRad(cur.theta);
        float errForward = dx * std::sin(curRad) + dy * std::cos(curRad);
        float errStrafe = dx * std::cos(curRad) - dy * std::sin(curRad);

        float headErr = lemlib::radToDeg(std::remainder(headingRad - curRad, 2.0 * M_PI));
        float turnCmd = std::clamp(headErr * 2.8f, -60.0f, 60.0f);
        float strafeCmd = std::clamp(errStrafe * 7.5f, -maxSpeed, maxSpeed);
        float forwardCmd = std::clamp(errForward * 7.5f, -maxSpeed, maxSpeed);

        holonomicDrive(forwardCmd, strafeCmd, turnCmd);
        pros::delay(10);
    }
    holonomicBrake();
}

// ============================================================================
// 5. Tuning & Diagnostic Test Suite
// ============================================================================

/**
 * @brief Test 1: Motor Polarity & Individual Motor Check
 * Tests all 4 motors individually for 1.2s each to verify wiring and polarity.
 */
void testMotorDirections() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  TEST: MOTOR DIRECTION & POLARITY CHECK  " << std::endl;
    std::cout << "==========================================" << std::endl;

    auto testOneMotor = [](const char* name, int port, pros::Motor& motor) {
        std::cout << "[Motor Test] Testing: " << name << " (Port " << port << ")..." << std::endl;
        controller.print(0, 0, "Test: %s P%d   ", name, port);
        motor.move(60);
        pros::delay(1200);
        motor.brake();
        pros::delay(300);
    };

    // 1. Test each motor individually (each wheel should spin FORWARD)
    testOneMotor("Front-Left", -11, frontLeft);
    testOneMotor("Back-Left", -20, backLeft);
    testOneMotor("Front-Right", 1, frontRight);
    testOneMotor("Back-Right", 10, backRight);

    // 2. Holistic tests
    std::cout << "[Holistic] Forward 1 sec..." << std::endl;
    controller.print(0, 0, "Move: Forward   ");
    holonomicDrive(60, 0, 0);
    pros::delay(1000);
    holonomicBrake();
    pros::delay(300);

    std::cout << "[Holistic] Strafe Right 1 sec..." << std::endl;
    controller.print(0, 0, "Move: Strafe R  ");
    holonomicDrive(0, 60, 0);
    pros::delay(1000);
    holonomicBrake();
    pros::delay(300);

    std::cout << "[Holistic] Spin Clockwise 1 sec..." << std::endl;
    controller.print(0, 0, "Move: Spin CW   ");
    holonomicDrive(0, 0, 60);
    pros::delay(1000);
    holonomicBrake();

    std::cout << "Motor Test Complete!" << std::endl;
    controller.print(0, 0, "Motor Check Done");
    controller.rumble(".");
}

/**
 * @brief Test 2: Track Width Spin Calibration (360 degrees)
 * Calculates the exact track width correction factor based on spin angle.
 */
void testTrackWidth(int fullRotations = 1) {
    float targetAngle = 360.0f * fullRotations;
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  TEST: TRACK WIDTH 360-DEGREE SPIN TEST  " << std::endl;
    std::cout << "  Current Track Width: " << TRACK_WIDTH_INCHES << " inches" << std::endl;
    std::cout << "==========================================" << std::endl;

    controller.print(0, 0, "Spinning 360... ");
    chassis.setPose(0, 0, 0);

    for (int i = 0; i < fullRotations; i++) {
        chassis.turnToHeading(180.0f, 2500, {.direction = lemlib::AngularDirection::CW_CLOCKWISE, .minSpeed = 40, .earlyExitRange = 15});
        chassis.turnToHeading(0.0f, 2500, {.direction = lemlib::AngularDirection::CW_CLOCKWISE});
        chassis.waitUntilDone();
    }

    lemlib::Pose endPose = chassis.getPose();
    float measuredAngle = endPose.theta;
    if (std::abs(measuredAngle) < 1.0f) measuredAngle += targetAngle; // Handle wrap-around

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Target Angle:    " << targetAngle << " deg" << std::endl;
    std::cout << "LemLib Odometry: " << measuredAngle << " deg" << std::endl;
    std::cout << "\n--- TUNING INSTRUCTIONS ---" << std::endl;
    std::cout << "1. Look at the physical robot on the field tiles:" << std::endl;
    std::cout << "   - If it UNDER-ROTATED (didn't make a full 360° circle):" << std::endl;
    std::cout << "     -> INCREASE TRACK_WIDTH_INCHES in main.cpp!" << std::endl;
    std::cout << "   - If it OVER-ROTATED (spun more than a full 360° circle):" << std::endl;
    std::cout << "     -> DECREASE TRACK_WIDTH_INCHES in main.cpp!" << std::endl;
    std::cout << "2. Formula: New_Track_Width = Old_Track_Width * (360.0 / Actual_Physical_Degrees)" << std::endl;
    std::cout << "---------------------------" << std::endl;

    controller.print(0, 0, "Spin Done! Check field");
    controller.rumble(".");
}

/**
 * @brief Test 3: Linear Forward Distance Test (24 inches)
 * Measures linear PID / LQR settling time, overshoot, and error.
 */
void testLinearDrive(float targetInches = 24.0f) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  TEST: LINEAR DRIVE (" << targetInches << " inches)      " << std::endl;
    std::cout << "==========================================" << std::endl;

    controller.print(0, 0, "Driving %2.0fin... ", targetInches);
    chassis.setPose(0, 0, 0);

    uint32_t startTime = pros::millis();
    chassis.moveToPoint(0, targetInches, 3000);
    chassis.waitUntilDone();
    uint32_t elapsed = pros::millis() - startTime;

    lemlib::Pose endPose = chassis.getPose();
    float errorY = targetInches - endPose.y;
    float driftX = endPose.x;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Target Y:     " << targetInches << " in" << std::endl;
    std::cout << "Final Y:      " << endPose.y << " in" << std::endl;
    std::cout << "Error Y:      " << errorY << " in" << std::endl;
    std::cout << "Drift X:      " << driftX << " in" << std::endl;
    std::cout << "Settle Time:  " << elapsed << " ms" << std::endl;

    controller.print(0, 0, "Err: %+4.2fin %4dms", errorY, (int)elapsed);
    controller.rumble(".");
}

/**
 * @brief Test 4: Angular Snap Turn Test (90 degrees)
 * Measures heading accuracy, steady-state error, and damping.
 */
void testAngularTurn(float targetHeading = 90.0f) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  TEST: ANGULAR TURN (" << targetHeading << " degrees)    " << std::endl;
    std::cout << "==========================================" << std::endl;

    controller.print(0, 0, "Turning %3.0fdeg... ", targetHeading);
    chassis.setPose(0, 0, 0);

    uint32_t startTime = pros::millis();
    int timeout = std::max(2200, static_cast<int>(std::abs(targetHeading) * 16));
    chassis.turnToHeading(targetHeading, timeout);
    chassis.waitUntilDone();
    uint32_t elapsed = pros::millis() - startTime;

    lemlib::Pose endPose = chassis.getPose();
    float errorTheta = targetHeading - endPose.theta;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Target Heading:  " << targetHeading << " deg" << std::endl;
    std::cout << "Final Heading:   " << endPose.theta << " deg" << std::endl;
    std::cout << "Heading Error:   " << errorTheta << " deg" << std::endl;
    std::cout << "Settle Time:     " << elapsed << " ms" << std::endl;

    controller.print(0, 0, "Err: %+4.1fdeg %4dms", errorTheta, (int)elapsed);
    controller.rumble(".");
}

/**
 * @brief Test 5: Lateral Strafe Test (24 inches sideways)
 * Measures sideways motion accuracy and heading retention.
 */
void testLateralStrafe(float strafeInches = 24.0f) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  TEST: LATERAL STRAFE (" << strafeInches << " inches)    " << std::endl;
    std::cout << "==========================================" << std::endl;

    controller.print(0, 0, "Strafe %2.0fin...  ", strafeInches);
    chassis.setPose(0, 0, 0);

    uint32_t startTime = pros::millis();
    holonomicStrafe(strafeInches, 0.0f, 3000, 100.0f);
    uint32_t elapsed = pros::millis() - startTime;

    lemlib::Pose endPose = chassis.getPose();
    float errorX = strafeInches - endPose.x;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Target Strafe: " << strafeInches << " in" << std::endl;
    std::cout << "Final Pose:    X=" << endPose.x << ", Y=" << endPose.y << ", Th=" << endPose.theta << std::endl;
    std::cout << "Strafe Error:  " << errorX << " in" << std::endl;
    std::cout << "Settle Time:   " << elapsed << " ms" << std::endl;

    controller.print(0, 0, "StrErr:%+4.1fin %3d", errorX, (int)elapsed);
    controller.rumble(".");
}

/**
 * @brief Test 6: Feedforward Friction Characterization (kS)
 * Determines static voltage threshold needed to overcome stiction.
 */
void testFeedforwardCharacterization() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  TEST: FEEDFORWARD kS CHARACTERIZATION   " << std::endl;
    std::cout << "==========================================" << std::endl;

    controller.print(0, 0, "Measuring kS... ");
    double found_kS = 0.0;

    for (int mv = 100; mv <= 4000; mv += 50) {
        leftMotors.move_voltage(mv);
        rightMotors.move_voltage(mv);
        pros::delay(80);

        double velLeft = std::abs(leftMotors.get_actual_velocity(0));
        double velRight = std::abs(rightMotors.get_actual_velocity(0));

        if (velLeft > 4.0 || velRight > 4.0) {
            found_kS = mv / 1000.0;
            std::cout << ">>> MEASURED kS: " << found_kS << " Volts <<<" << std::endl;
            break;
        }
    }
    leftMotors.brake();
    rightMotors.brake();

    controller.print(0, 0, "kS = %4.2f Volts   ", found_kS);
    controller.rumble("-");
}

/**
 * @brief Helper: Point structure for 2D spline trajectory
 */
struct Point2D {
    float x;
    float y;
};

/**
 * @brief Helper: Evaluate cubic Bézier curve at parameter t in [0, 1]
 */
inline Point2D bezierPoint(Point2D p0, Point2D p1, Point2D p2, Point2D p3, float t) {
    float u = 1.0f - t;
    float tt = t * t;
    float uu = u * u;
    float uuu = uu * u;
    float ttt = tt * t;

    Point2D p;
    p.x = uuu * p0.x + 3.0f * uu * t * p1.x + 3.0f * u * tt * p2.x + ttt * p3.x;
    p.y = uuu * p0.y + 3.0f * uu * t * p1.y + 3.0f * u * tt * p2.y + ttt * p3.y;
    return p;
}

/**
 * @brief Test 7: Drive Forward 24" while simultaneously turning 180° (Button LEFT)
 * Holonomic simultaneous 3-DOF translation and rotation.
 */
void testForwardTurn180(float targetDist = 24.0f, float targetHeading = 180.0f, int timeout = 3500) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  TEST: FORWARD " << targetDist << "\" + TURN " << targetHeading << "° SIMULTANEOUSLY" << std::endl;
    std::cout << "==========================================" << std::endl;

    controller.print(0, 0, "Fwd24+Turn180... ");
    chassis.setPose(0, 0, 0);

    uint32_t startTime = pros::millis();
    uint32_t settledTime = 0;
    float prevFwd = 0.0f, prevStrf = 0.0f, prevTurn = 0.0f;

    while (pros::millis() - startTime < static_cast<uint32_t>(timeout)) {
        lemlib::Pose cur = chassis.getPose();

        // Global position errors
        float errGlobalX = 0.0f - cur.x;
        float errGlobalY = targetDist - cur.y;
        float distErr = std::hypot(errGlobalX, errGlobalY);

        // Heading error (shortest angular distance to 180°)
        float headErrDeg = lemlib::radToDeg(lemlib::angleError(lemlib::degToRad(targetHeading), lemlib::degToRad(cur.theta)));

        // Settle check
        if (distErr < 0.45f && std::abs(headErrDeg) < 0.8f) {
            if (settledTime == 0) settledTime = pros::millis();
            else if (pros::millis() - settledTime > 150) break;
        } else {
            settledTime = 0;
        }

        // Transform global error into robot-centric local frame based on current heading
        float curRad = lemlib::degToRad(cur.theta);
        float errForward = errGlobalX * std::sin(curRad) + errGlobalY * std::cos(curRad);
        float errStrafe  = errGlobalX * std::cos(curRad) - errGlobalY * std::sin(curRad);

        // Local velocities for LQR damping
        float velForward = lemlib::getLocalSpeed(true).y;
        float velStrafe  = lemlib::getLocalSpeed(true).x;
        float velTurn    = lemlib::getLocalSpeed().theta;

        // Stiction feedforward
        float stictionFwd = (std::abs(errForward) > 0.15f) ? std::clamp(errForward / 1.0f, -1.0f, 1.0f) * 11.0f : 0.0f;
        float stictionStr = (std::abs(errStrafe) > 0.15f)  ? std::clamp(errStrafe / 1.0f, -1.0f, 1.0f) * 11.0f : 0.0f;
        float stictionTurn = (std::abs(headErrDeg) > 0.15f) ? std::clamp(headErrDeg / 1.2f, -1.0f, 1.0f) * 8.5f : 0.0f;

        // Control outputs (PID + LQR state velocity damping + stiction)
        float cmdFwd  = errForward * 8.5f - 0.40f * velForward + stictionFwd;
        float cmdStr  = errStrafe * 8.5f - 0.40f * velStrafe + stictionStr;
        float cmdTurn = headErrDeg * 3.2f - 0.18f * velTurn + stictionTurn;

        // Cap speeds and slew rate limit
        cmdFwd = std::clamp(cmdFwd, -90.0f, 90.0f);
        cmdStr = std::clamp(cmdStr, -90.0f, 90.0f);
        cmdTurn = std::clamp(cmdTurn, -70.0f, 70.0f);

        cmdFwd = lemlib::slew(cmdFwd, prevFwd, 15.0f);
        cmdStr = lemlib::slew(cmdStr, prevStrf, 15.0f);
        cmdTurn = lemlib::slew(cmdTurn, prevTurn, 15.0f);

        prevFwd = cmdFwd;
        prevStrf = cmdStr;
        prevTurn = cmdTurn;

        holonomicDrive(cmdFwd, cmdStr, cmdTurn);
        pros::delay(10);
    }

    holonomicBrake();
    uint32_t elapsed = pros::millis() - startTime;

    lemlib::Pose endPose = chassis.getPose();
    float errY = targetDist - endPose.y;
    float errTh = lemlib::radToDeg(lemlib::angleError(lemlib::degToRad(targetHeading), lemlib::degToRad(endPose.theta)));

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Target: Y=" << targetDist << "in, Theta=" << targetHeading << "deg" << std::endl;
    std::cout << "Final:  X=" << endPose.x << ", Y=" << endPose.y << ", Theta=" << endPose.theta << "deg" << std::endl;
    std::cout << "Error:  Y=" << errY << "in, Theta=" << errTh << "deg (" << elapsed << "ms)" << std::endl;

    controller.print(0, 0, "Y:%+3.1f Th:%+3.1f", errY, errTh);
    controller.rumble(".");
}

/**
 * @brief Test 8: Drive Forward 24" and Strafe Right 24" along Hypotenuse (Button DOWN)
 * Moves along straight 45-degree diagonal line while holding heading locked at 0 degrees.
 */
void testHypotenuseDrive(float targetX = 24.0f, float targetY = 24.0f, int timeout = 3500) {
    float hypotDist = std::hypot(targetX, targetY);
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  TEST: HYPOTENUSE DRIVE (X=" << targetX << "\", Y=" << targetY << "\", Hyp=" << hypotDist << "\")" << std::endl;
    std::cout << "==========================================" << std::endl;

    controller.print(0, 0, "Hypotenuse 24\"... ");
    chassis.setPose(0, 0, 0);

    uint32_t startTime = pros::millis();
    uint32_t settledTime = 0;
    float prevFwd = 0.0f, prevStrf = 0.0f, prevTurn = 0.0f;

    while (pros::millis() - startTime < static_cast<uint32_t>(timeout)) {
        lemlib::Pose cur = chassis.getPose();

        float errGlobalX = targetX - cur.x;
        float errGlobalY = targetY - cur.y;
        float distErr = std::hypot(errGlobalX, errGlobalY);

        // Lock heading at 0 degrees
        float headErrDeg = lemlib::radToDeg(lemlib::angleError(0.0f, lemlib::degToRad(cur.theta)));

        if (distErr < 0.45f && std::abs(headErrDeg) < 0.6f) {
            if (settledTime == 0) settledTime = pros::millis();
            else if (pros::millis() - settledTime > 150) break;
        } else {
            settledTime = 0;
        }

        float curRad = lemlib::degToRad(cur.theta);
        float errForward = errGlobalX * std::sin(curRad) + errGlobalY * std::cos(curRad);
        float errStrafe  = errGlobalX * std::cos(curRad) - errGlobalY * std::sin(curRad);

        float velForward = lemlib::getLocalSpeed(true).y;
        float velStrafe  = lemlib::getLocalSpeed(true).x;
        float velTurn    = lemlib::getLocalSpeed().theta;

        float stictionFwd = (std::abs(errForward) > 0.15f) ? std::clamp(errForward / 1.0f, -1.0f, 1.0f) * 11.0f : 0.0f;
        float stictionStr = (std::abs(errStrafe) > 0.15f)  ? std::clamp(errStrafe / 1.0f, -1.0f, 1.0f) * 11.0f : 0.0f;
        float stictionTurn = (std::abs(headErrDeg) > 0.15f) ? std::clamp(headErrDeg / 1.2f, -1.0f, 1.0f) * 8.5f : 0.0f;

        float cmdFwd  = errForward * 8.5f - 0.40f * velForward + stictionFwd;
        float cmdStr  = errStrafe * 8.5f - 0.40f * velStrafe + stictionStr;
        float cmdTurn = headErrDeg * 3.2f - 0.18f * velTurn + stictionTurn;

        cmdFwd = std::clamp(cmdFwd, -95.0f, 95.0f);
        cmdStr = std::clamp(cmdStr, -95.0f, 95.0f);
        cmdTurn = std::clamp(cmdTurn, -50.0f, 50.0f);

        cmdFwd = lemlib::slew(cmdFwd, prevFwd, 15.0f);
        cmdStr = lemlib::slew(cmdStr, prevStrf, 15.0f);
        cmdTurn = lemlib::slew(cmdTurn, prevTurn, 15.0f);

        prevFwd = cmdFwd;
        prevStrf = cmdStr;
        prevTurn = cmdTurn;

        holonomicDrive(cmdFwd, cmdStr, cmdTurn);
        pros::delay(10);
    }

    holonomicBrake();
    uint32_t elapsed = pros::millis() - startTime;

    lemlib::Pose endPose = chassis.getPose();
    float errX = targetX - endPose.x;
    float errY = targetY - endPose.y;
    float errDist = std::hypot(errX, errY);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Target: X=" << targetX << "in, Y=" << targetY << "in" << std::endl;
    std::cout << "Final:  X=" << endPose.x << ", Y=" << endPose.y << ", Theta=" << endPose.theta << "deg" << std::endl;
    std::cout << "Error:  Dist=" << errDist << "in (dX=" << errX << ", dY=" << errY << ") in " << elapsed << "ms" << std::endl;

    controller.print(0, 0, "HypErr:%+3.1fin %3d", errDist, (int)elapsed);
    controller.rumble(".");
}

/**
 * @brief Test 9: Drive to (24, 24) along a smooth Cubic Spline using Pedro Pathing (Button RIGHT)
 * Follows a smooth curved trajectory using reactive vector fields (GVF), predictive braking,
 * centripetal feedforward, and decoupled heading hold.
 */
void testSplineDrive(float targetX = 24.0f, float targetY = 24.0f, int timeout = 4000) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  TEST: PEDRO PATHING BÉZIER VECTOR FOLLOWER TO (X=" << targetX << "\", Y=" << targetY << "\")" << std::endl;
    std::cout << "==========================================" << std::endl;

    controller.print(0, 0, "Pedro 24\"x24\"... ");
    chassis.setPose(0, 0, 0);

    // 1. Define Cubic Bézier curve control points:
    // Leaves (0, 0) moving forward along +Y, smoothly arcs right to (targetX, targetY)
    pedro::Point p0(0.0f, 0.0f);
    pedro::Point p1(0.0f, targetY * 0.65f);
    pedro::Point p2(targetX * 0.60f, targetY);
    pedro::Point p3(targetX, targetY);
    pedro::BezierCurve curve(p0, p1, p2, p3);

    // 2. Wrap in PedroPath with Decoupled Heading (Hold 0 degrees heading throughout the curve)
    pedro::PedroPath path(curve, pedro::HeadingMode::CONSTANT, 0.0f);

    // 3. Pedro Pathing Vector Field Follower
    pedro::PedroFollower follower;
    uint32_t startTime = pros::millis();

    follower.follow(
        path,
        timeout,
        [](int forward, int strafe, int turn) {
            holonomicDrive(forward, strafe, turn);
        },
        []() {
            holonomicBrake();
        }
    );

    uint32_t elapsed = pros::millis() - startTime;
    lemlib::Pose endPose = chassis.getPose();
    float errX = targetX - endPose.x;
    float errY = targetY - endPose.y;
    float errDist = std::hypot(errX, errY);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Pedro Target: X=" << targetX << "in, Y=" << targetY << "in" << std::endl;
    std::cout << "Final Pose:   X=" << endPose.x << ", Y=" << endPose.y << ", Theta=" << endPose.theta << "deg" << std::endl;
    std::cout << "Pedro Error:  Dist=" << errDist << "in (dX=" << errX << ", dY=" << errY << ") in " << elapsed << "ms" << std::endl;

    controller.print(0, 0, "PedErr:%+3.1fin %3d", errDist, (int)elapsed);
    controller.rumble(".");
}

// ============================================================================
// 6. PROS Lifecycle Functions
// ============================================================================

void initialize() {
    pros::lcd::initialize();
    chassis.calibrate();

    // Set drivebase to X-Drive to enable 4-motor holonomic kinematic odometry (tracks both Y and sideways X)
    chassis.setDrivebaseType(lemlib::DrivebaseType::XDRIVE);

    // Enable unified HYBRID mode (LQR optimal state velocity damping + PID setpoint tracking)
    chassis.useHybrid();

    // Start background health watchdog (500ms period)
    motorMonitor.startTask(500);

    // Background Telemetry Task for Brain LCD (20 Hz)
    pros::Task telemetryTask([&]() {
        while (true) {
            lemlib::Pose pose = chassis.getPose();

            pros::lcd::print(0, "X-DRIVE TUNING SUITE [HYBRID LQR+PID]");
            pros::lcd::print(1, "Pose: X:%5.1f Y:%5.1f Th:%5.1f", pose.x, pose.y, pose.theta);
            pros::lcd::print(2, "FL:%2.0fC BL:%2.0fC FR:%2.0fC BR:%2.0fC",
                             frontLeft.get_temperature(), backLeft.get_temperature(),
                             frontRight.get_temperature(), backRight.get_temperature());
            pros::lcd::print(3, "Battery: %2.0f%% | %4.2f V", (double)pros::battery::get_capacity(), pros::battery::get_voltage() / 1000.0);
            pros::lcd::print(4, "[A]=360 [B]=24in [Y]=90deg [X]=Strf");
            pros::lcd::print(5, "[L]=24+180 [D]=Hypot [R]=Spline");

            pros::delay(50);
        }
    });

    std::cout << "\n[OK] X-Drive Tuning Program Initialized [HYBRID LQR+PID Mode Active]." << std::endl;
}

void disabled() {}

void competition_initialize() {}

void autonomous() {
    // When connected to a competition switch, running autonomous triggers 24" drive test
    testLinearDrive(24.0f);
}

// ============================================================================
// 7. Interactive Driver Control & Tuning Test Trigger Loop
// ============================================================================

void opcontrol() {
    controller.print(0, 0, "X-Drive Suite Ready ");

    while (true) {
        // --- BUTTON UP: Motor Direction & Polarity Diagnostic ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_UP)) {
            testMotorDirections();
        }

        // --- BUTTON A: 360-Degree Track Width Calibration Spin Test ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_A)) {
            testTrackWidth(1);
        }

        // --- BUTTON B: 24-inch Linear Drive Test ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_B)) {
            testLinearDrive(24.0f);
        }

        // --- BUTTON Y: 90-Degree Angular Turn Snap Test ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_Y)) {
            testAngularTurn(90.0f);
        }

        // --- BUTTON X: 24-inch Lateral Strafe Test ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_X)) {
            testLateralStrafe(24.0f);
        }

        // --- BUTTON LEFT: Forward 24" while rotating 180° simultaneously ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_LEFT)) {
            testForwardTurn180(24.0f, 180.0f);
        }

        // --- BUTTON DOWN: Forward 24" and Strafe Right 24" along Hypotenuse ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_DOWN)) {
            testHypotenuseDrive(24.0f, 24.0f);
        }

        // --- BUTTON RIGHT: Cubic Spline Trajectory to (24", 24") ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_RIGHT)) {
            testSplineDrive(24.0f, 24.0f);
        }

        // --- BUTTON L1: Reset Odometry Pose to (0, 0, 0) ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_L1)) {
            chassis.setPose(0, 0, 0);
            controller.print(0, 0, "Pose Reset (0,0,0)  ");
            controller.rumble(".");
        }

        // --- BUTTON R1: Feedforward Friction kS Characterization ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_R1)) {
            testFeedforwardCharacterization();
        }

        // ====================================================================
        // Holonomic 3-DOF Driving Control (Reversed Forward/Back & Turn):
        // Left Stick Y (Axis 3) = Inverted (-Y): Forward is Back, Back is Forward
        // Left Stick X (Axis 4) = Lateral Strafe Left / Right
        // Right Stick X (Axis 1) = Inverted (-X): Turn direction reversed
        // ====================================================================
        int forward = -controller.get_analog(pros::E_CONTROLLER_ANALOG_LEFT_Y);
        int strafe  = controller.get_analog(pros::E_CONTROLLER_ANALOG_LEFT_X);
        int turn    = -controller.get_analog(pros::E_CONTROLLER_ANALOG_RIGHT_X);

        // Deadband filter (ignore stick drift below threshold)
        if (std::abs(forward) < 5) forward = 0;
        if (std::abs(strafe) < 5)  strafe = 0;
        if (std::abs(turn) < 5)    turn = 0;

        holonomicDrive(forward, strafe, turn);

        pros::delay(10);
    }
}
