#include "main.h"
#include "lemlib/api.hpp" // IWYU pragma: keep
#include "subsystems/flc/FuzzyLogic.hpp"
#include "subsystems/ekf/EKF.hpp"
#include "subsystems/trajectory/QuinticSpline.hpp"
#include <iostream>
#include <cmath>
#include <vector>

// ============================================================================
// 1. Controller & Hardware Configuration
// ============================================================================
pros::Controller controller(pros::E_CONTROLLER_MASTER);

// Motor groups (6-motor drive, 200 RPM green cartridges)
// Configured for physical motor wiring and polarity
pros::MotorGroup leftMotors({-3, 18, -5}, pros::MotorGearset::blue);
pros::MotorGroup rightMotors({-10, 3, -17}, pros::MotorGearset::blue);

// Optional mechanism motors
pros::MotorGroup intakeMotors({2}, pros::MotorGearset::blue);

// Sensors for Odometry & Localization (set to nullptr for base-only)
// pros::Imu imu(21);

// Drivetrain geometry and kinematics
// TUNE THIS: Adjust trackWidth if 360-degree spin test under/over-rotates!
constexpr float TRACK_WIDTH_INCHES = 10.5f;
constexpr float WHEEL_DIAMETER_INCHES = 3.25f;
constexpr float DRIVETRAIN_RPM = 450.0f;

lemlib::Drivetrain drivetrain(&leftMotors, // left motor group
                              &rightMotors, // right motor group
                              TRACK_WIDTH_INCHES,
                              lemlib::Omniwheel::NEW_325, // 3.25-inch wheels
                              DRIVETRAIN_RPM,
                              2.0f);

// ============================================================================
// 2. Motion Controller Configurations (PID + LQR + FLC + EKF)
// ============================================================================

// Lateral PID Controller
lemlib::ControllerSettings linearController(10.0f, // kP: increase if stops short, decrease if oscillates
                                            0.0f, // kI: integral gain for steady-state error
                                            3.0f, // kD: derivative damping
                                            3.0f, // anti windup
                                            1.0f, // small error range (in)
                                            100, // small error timeout (ms)
                                            3.0f, // large error range (in)
                                            500, // large error timeout (ms)
                                            20.0f // slew rate limit
);

// Angular PID Controller
lemlib::ControllerSettings angularController(2.0f, // kP: turn stiffness
                                             0.0f, // kI
                                             10.0f, // kD: turn damping
                                             3.0f, // anti windup
                                             1.0f, // small error range (deg)
                                             100, // small error timeout (ms)
                                             3.0f, // large error range (deg)
                                             500, // large error timeout (ms)
                                             0.0f // slew rate limit
);

// Lateral LQR Controller (Optimal State Feedback)
lemlib::LQRSettings lateralLQR(10.0f, // kP: position error gain
                               3.2f, // kV: velocity state feedback damping (counters momentum)
                               0.0f, // kA: 0 (no IMU accelerometer)
                               0.0f, // kI
                               3.0f, // anti windup
                               1.0f, // small error (in)
                               100, // small error timeout (ms)
                               3.0f, // large error (in)
                               500, // large error timeout (ms)
                               20.0f, // slew limit
                               false // use IMU prediction
);

// Angular LQR Controller (Optimal Heading Control)
lemlib::LQRSettings angularLQR(1.0f, // kP: angular position gain
                               1.8f, // kV: angular velocity damping (eliminates wobble on snap turns)
                               0.0f, // kA: 0 (no IMU gyro)
                               0.0f, // kI
                               3.0f, // anti windup
                               1.0f, // small error (deg)
                               100, // small error timeout (ms)
                               3.0f, // large error (deg)
                               500, // large error timeout (ms)
                               0.0f, // slew limit
                               false // use IMU prediction
);

// Odometry Sensors Struct (All nullptr -> uses motor encoders)
lemlib::OdomSensors sensors(nullptr, nullptr, nullptr, nullptr, nullptr);

// Exponential Drive Curves for driver control
lemlib::ExpoDriveCurve throttleCurve(3, 10, 1.019);
lemlib::ExpoDriveCurve steerCurve(3, 10, 1.019);

// LemLib Chassis Instance
lemlib::Chassis chassis(drivetrain, linearController, angularController, lateralLQR, angularLQR, sensors,
                        &throttleCurve, &steerCurve);

// ============================================================================
// 3. Subsystem Integrations: LTV, EKF, FLC, TCS & Watchdogs
// ============================================================================

VelocityControllerConfig velConfig {.kV = 6.2,
                                    .KA_straight = 0.25,
                                    .KA_turn = 0.2,
                                    .KS_straight = 0.45,
                                    .KS_turn = 0.35,
                                    .KP_straight = 2.0,
                                    .KI_straight = 0.0,
                                    .max_voltage = 12.0,
                                    .trackWidthMeters = TRACK_WIDTH_INCHES * lemlib::INCH_TO_METER,
                                    .enableTCS = true,
                                    .maxSlipRatio = 0.18};

lemlib::LTVPathFollower ltvFollower(chassis, leftMotors, rightMotors, velConfig);

// 5-State Extended Kalman Filter instance
lemlib::RobotEKF robotEKF(lemlib::Pose(0, 0, 0));

// Fuzzy Logic Adaptive Controllers
lemlib::FuzzyLogicController lateralFLC(24.0, 40.0);
lemlib::FuzzyLogicController angularFLC(90.0, 180.0);

lemlib::MotorMonitor
    motorMonitor(controller, {{"LeftDrive", &leftMotors}, {"RightDrive", &rightMotors}, {"Intake", &intakeMotors}});

lemlib::ColorSorter colorSorter(nullptr, &intakeMotors, lemlib::AllianceColor::RED);

ASSET(example_txt);

// ============================================================================
// 4. Interactive Tuning & Calibration Test Suite
// ============================================================================

/**
 * @brief TEST 1: Track Width Calibration (360-Degree Spin Test)
 * Rotates exactly 360 degrees based on motor encoders.
 * If robot turns LESS than 360 on the field -> DECREASE TRACK_WIDTH_INCHES.
 * If robot turns MORE than 360 on the field -> INCREASE TRACK_WIDTH_INCHES.
 */
void testTrackWidth(int fullRotations = 1) {
    std::cout << "\n=== STARTING TRACK WIDTH CALIBRATION ===" << std::endl;
    controller.print(0, 0, "Test: Track Width...");
    chassis.setPose(0, 0, 0);

    float targetHeading = 360.0f * fullRotations;
    chassis.turnToHeading(targetHeading, 3000 * fullRotations);
    chassis.waitUntilDone();

    lemlib::Pose endPose = chassis.getPose();
    std::cout << "Target Angle:   " << targetHeading << " deg" << std::endl;
    std::cout << "Measured Angle: " << endPose.theta << " deg" << std::endl;
    std::cout << "-> If robot OVER-turned on field, INCREASE TRACK_WIDTH_INCHES" << std::endl;
    std::cout << "-> If robot UNDER-turned on field, DECREASE TRACK_WIDTH_INCHES" << std::endl;
    controller.print(0, 0, "Done: %5.1f deg  ", endPose.theta);
    controller.rumble(".");
}

/**
 * @brief TEST 2: Linear Drive Test (24 or 48 inches)
 * Tests position accuracy, overshoot, and settle time.
 */
void testLinearDrive(float targetInches = 24.0f) {
    std::cout << "\n=== STARTING LINEAR DRIVE TEST (" << targetInches << " in) ===" << std::endl;
    controller.print(0, 0, "Test: Lin %3.0fin...", targetInches);
    chassis.setPose(0, 0, 0);

    uint32_t startTime = pros::millis();
    chassis.moveToPoint(0, targetInches, 3000);
    chassis.waitUntilDone();
    uint32_t elapsed = pros::millis() - startTime;

    lemlib::Pose endPose = chassis.getPose();
    float error = targetInches - endPose.y;
    std::cout << "Final Y: " << endPose.y << " in | Error: " << error << " in | Settle Time: " << elapsed << " ms"
              << std::endl;
    controller.print(0, 0, "Err: %4.2fin %4dms", error, (int)elapsed);
    controller.rumble(".");
}

/**
 * @brief TEST 3: Angular Snap Test (90-Degree Turn)
 * Tests heading stiffness (kP) and damping (kV / kD).
 */
void testAngularTurn(float targetHeading = 90.0f) {
    std::cout << "\n=== STARTING ANGULAR TURN TEST (" << targetHeading << " deg) ===" << std::endl;
    controller.print(0, 0, "Test: Turn %3.0fdeg", targetHeading);
    chassis.setPose(0, 0, 0);

    uint32_t startTime = pros::millis();
    chassis.turnToHeading(targetHeading, 1500);
    chassis.waitUntilDone();
    uint32_t elapsed = pros::millis() - startTime;

    lemlib::Pose endPose = chassis.getPose();
    float error = targetHeading - endPose.theta;
    std::cout << "Final Theta: " << endPose.theta << " deg | Error: " << error << " deg | Settle: " << elapsed << " ms"
              << std::endl;
    controller.print(0, 0, "Err: %4.1fdeg %4dms", error, (int)elapsed);
    controller.rumble(".");
}

/**
 * @brief TEST 4: LTV S-Curve Trajectory Tracking Test
 * Generates an on-the-fly 2-meter smooth curved trajectory and tracks it with LTV.
 */
void testLTVTrajectory() {
    std::cout << "\n=== STARTING LTV S-CURVE TRAJECTORY TEST ===" << std::endl;
    controller.print(0, 0, "Test: LTV S-Curve...");
    chassis.setPose(0, 0, 0);

    // Generate smooth 100-step S-curve path
    std::vector<lemlib::State> path;
    int steps = 100;
    double dt = 0.02; // 20ms step
    for (int i = 0; i <= steps; ++i) {
        double t = (double)i / steps;
        double x = 0.4 * std::sin(M_PI * t); // lateral displacement
        double y = 1.8 * t; // forward progress
        double v = 0.9 * std::sin(M_PI * t); // smooth bell-curve velocity
        double heading = std::atan2(0.4 * M_PI * std::cos(M_PI * t), 1.8);
        double w = (i > 0) ? (heading - path.back().heading) / dt : 0.0;
        path.push_back({x, y, heading, v, w});
    }

    ltvFollower.followTrajectory(path, {.log = true});
    ltvFollower.waitUntilDone();

    std::cout << "LTV S-Curve execution complete!" << std::endl;
    controller.print(0, 0, "LTV Done!          ");
    controller.rumble(".");
}

/**
 * @brief TEST 5: Static Friction (kS) & Max Velocity (kV) Measurement
 * Gradually ramps voltage to find the exact minimum voltage required to start moving.
 */
void testFeedforwardCharacterization() {
    std::cout << "\n=== STARTING FEEDFORWARD CHARACTERIZATION ===" << std::endl;
    controller.print(0, 0, "Calibrating kS...");

    // Ramp voltage from 0 to 4.0V slowly
    double found_kS = 0.0;
    for (int mv = 100; mv <= 4000; mv += 50) {
        leftMotors.move_voltage(mv);
        rightMotors.move_voltage(mv);
        pros::delay(80);

        double velLeft = std::abs(leftMotors.get_actual_velocity(0));
        double velRight = std::abs(rightMotors.get_actual_velocity(0));

        if (velLeft > 5.0 || velRight > 5.0) {
            found_kS = mv / 1000.0;
            std::cout << ">>> MEASURED kS (Static Friction Voltage): " << found_kS << " Volts" << std::endl;
            break;
        }
    }
    leftMotors.brake();
    rightMotors.brake();

    controller.print(0, 0, "kS = %4.2f Volts   ", found_kS);
    controller.rumble("-");
}

/**
 * @brief TEST 6: Smooth Jerk-Limited Quintic Spline S-Curve Generation
 * Computes a C^2 continuous 5th-order polynomial trajectory on-the-fly and tracks with LTV + TCS.
 */
void testQuinticSpline() {
    std::cout << "\n=== GENERATING JERK-LIMITED QUINTIC SPLINE TRAJECTORY ===" << std::endl;
    controller.print(0, 0, "Gen Spline 5th...");
    chassis.setPose(0, 0, 0);

    lemlib::QuinticSplineGenerator::SplineWaypoints params {
        .start = lemlib::Pose(0, 0, 0),
        .end = lemlib::Pose(20.0, 40.0, 45.0),
        .startVel = 0.0,
        .endVel = 0.0,
        .maxVel = 1.0,
        .maxAccel = 1.8,
        .maxJerk = 3.5
    };

    auto trajectory = lemlib::QuinticSplineGenerator::generateTrajectory(params, 0.01);
    std::cout << "Generated " << trajectory.size() << " trajectory states. Executing..." << std::endl;

    ltvFollower.followTrajectory(trajectory, {.log = true});
    ltvFollower.waitUntilDone();

    controller.print(0, 0, "Spline Done!       ");
    controller.rumble(".");
}

// ============================================================================
// 5. Lifecycle Functions
// ============================================================================

void initialize() {
    pros::lcd::initialize();
    chassis.calibrate();

    // Start background health watchdog
    motorMonitor.startTask(500);

    // EKF Fusion & Brain LCD telemetry task (100 Hz)
    pros::Task sensorFusionTask([&]() {
        uint32_t lastTime = pros::millis();
        while (true) {
            uint32_t now = pros::millis();
            double dt = (now - lastTime) / 1000.0;
            lastTime = now;

            // 1. EKF Kinematic Predict
            robotEKF.predict(dt);

            // 2. Fuse Odometry Pose
            lemlib::Pose rawPose = chassis.getPose(true);
            robotEKF.updatePose(rawPose.x * lemlib::INCH_TO_METER,
                                rawPose.y * lemlib::INCH_TO_METER,
                                rawPose.theta);

            // 3. Telemetry on Brain LCD
            lemlib::Pose fusedPose = robotEKF.getPose();
            pros::lcd::print(0, "EKF X: %5.1f in | Odom: %5.1f", fusedPose.x, rawPose.x);
            pros::lcd::print(1, "EKF Y: %5.1f in | Odom: %5.1f", fusedPose.y, rawPose.y);
            pros::lcd::print(2, "EKF Th: %5.1f deg", fusedPose.theta);
            pros::lcd::print(3, "TCS: ACTIVE | FLC: ON");
            pros::delay(10);
        }
    });
}

void disabled() {}

void competition_initialize() {}

// ============================================================================
// 6. Autonomous Routine
// ============================================================================

void autonomous() {
    // Default: Run smooth Quintic Spline trajectory or LQR drive
    chassis.useLQR();
    testLinearDrive(24.0f);
    testAngularTurn(90.0f);
}

// ============================================================================
// 7. Interactive Driver Control & Live Tuning Suite
// ============================================================================

void opcontrol() {
    while (true) {
        // --- BUTTON A: Run 360-Degree Track Width Spin Test ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_A)) { testTrackWidth(1); }

        // --- BUTTON B: Run 24-inch Linear Drive Test ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_B)) { testLinearDrive(24.0f); }

        // --- BUTTON Y: Run 90-Degree Turn Snap Test ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_Y)) { testAngularTurn(90.0f); }

        // --- BUTTON UP: Run LTV S-Curve Trajectory Test ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_UP)) { testLTVTrajectory(); }

        // --- BUTTON RIGHT: Run Jerk-Limited Quintic Spline Test ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_RIGHT)) { testQuinticSpline(); }

        // --- BUTTON DOWN: Run Feedforward kS / kV Characterization ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_DOWN)) { testFeedforwardCharacterization(); }

        // --- BUTTON X: Toggle Controller Mode (LQR <-> PID) ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_X)) {
            if (chassis.getMotionControllerType() == lemlib::MotionControllerType::PID) {
                chassis.useLQR();
                controller.print(0, 0, "Mode: LQR (Optimal) ");
                controller.rumble(".");
            } else {
                chassis.usePID();
                controller.print(0, 0, "Mode: PID (Classic) ");
                controller.rumble("-");
            }
        }

        // Normal driving control (Arcade Drive)
        int leftY = controller.get_analog(pros::E_CONTROLLER_ANALOG_LEFT_Y); // Axis 3 (Throttle)
        int rightX = controller.get_analog(pros::E_CONTROLLER_ANALOG_RIGHT_X); // Axis 1 (Turn)
        chassis.arcade(leftY, rightX);

        pros::delay(10);
    }
}
