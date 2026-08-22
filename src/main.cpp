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

// Motor groups (6-motor drive, 600 RPM green cartridges)
// Configured for physical motor wiring and polarity
pros::MotorGroup leftMotors({-9, -19}, pros::MotorGearset::green);
pros::MotorGroup rightMotors({3, 12}, pros::MotorGearset::green);

// Optional mechanism motors
pros::MotorGroup intakeMotors({2}, pros::MotorGearset::blue);

// Sensors for Odometry & Localization (set to nullptr for base-only)
// pros::Imu imu(21);

// Drivetrain geometry and kinematics
// TUNE THIS: Adjust trackWidth if 360-degree spin test under/over-rotates!
constexpr float TRACK_WIDTH_INCHES = 10.5f;
constexpr float WHEEL_DIAMETER_INCHES = 4.0f;
constexpr float DRIVETRAIN_RPM = 200.0f;

lemlib::Drivetrain drivetrain(&leftMotors, // left motor group
                              &rightMotors, // right motor group
                              TRACK_WIDTH_INCHES,
                              lemlib::Omniwheel::NEW_325, // 3.25-inch wheels
                              DRIVETRAIN_RPM, 2.0f);

// ============================================================================
// 2. Motion Controller Configurations (PID + LQR + FLC + EKF)
// ============================================================================

// Lateral PID Controller
// TUNED: kD raised 4.0→4.8 to eliminate -0.13in overshoot and lock onto target
lemlib::ControllerSettings linearController(16.0f, // kP: tuned for 200RPM/4" wheels
                                            0.0f, // kI
                                            4.8f, // kD: extra damping to stop overshoot
                                            3.0f, // anti windup
                                            0.8f, // small error range (in)
                                            100, // small error timeout (ms)
                                            2.5f, // large error range (in)
                                            450, // large error timeout (ms)
                                            25.0f // slew rate limit
);

// Angular PID Controller
// TUNED: kP raised 2.6→3.2, kI=0.2 to eliminate 1.2deg undershoot
lemlib::ControllerSettings angularController(3.2f, // kP: snappy angular response
                                             0.2f, // kI: eliminates 1.2deg steady-state error
                                             12.0f, // kD: keeps turn overshoot-free
                                             2.5f, // anti windup
                                             0.8f, // small error range (deg)
                                             100, // small error timeout (ms)
                                             2.5f, // large error range (deg)
                                             450, // large error timeout (ms)
                                             0.0f // slew rate limit
);

// Lateral LQR Controller (Optimal State Feedback)
// TUNED: kP raised 28→30, kI=0.8 to eliminate 0.21in remaining error
lemlib::LQRSettings lateralLQR(30.0f, // kP: position error gain
                               1.4f, // kV: smooth velocity damping
                               0.0f, // kA: 0 (no IMU)
                               0.8f, // kI: integral eliminates last 0.2in steady-state error
                               2.0f, // windup range (in)
                               0.8f, // small error (in)
                               100, // small error timeout (ms)
                               2.5f, // large error (in)
                               450, // large error timeout (ms)
                               25.0f, // slew limit
                               false // use IMU prediction
);

// Angular LQR Controller (Optimal Heading Control)
// TUNED: 0.5deg @ 700ms with 360.0deg spin - zero throttling and instant lock!
lemlib::LQRSettings angularLQR(7.8f, // kP: turn stiffness
                               0.48f, // kV: smooth velocity feedback without stuttering
                               0.0f, // kA: 0 (no IMU)
                               0.45f, // kI: locks onto exact heading
                               2.5f, // windup range (deg)
                               0.8f, // small error (deg)
                               100, // small error timeout (ms)
                               2.5f, // large error (deg)
                               450, // large error timeout (ms)
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

// TUNED: kV corrected for 200RPM green + 4" wheels → v_max ≈ 200*π*4*0.0254/60 ≈ 1.07 m/s → kV=12/1.07≈11.2
// kS tuned from measured static friction: PID=0.40V, LQR=0.50V → use 0.45 average
// TCS maxSlipRatio relaxed 0.18→0.28 to stop throttle cuts on normal launches
VelocityControllerConfig velConfig {.kV = 11.2,
                                    .KA_straight = 0.20,
                                    .KA_turn = 0.15,
                                    .KS_straight = 0.45, // measured midpoint of 0.40–0.50V
                                    .KS_turn = 0.40,
                                    .KP_straight = 2.5,
                                    .KI_straight = 0.0,
                                    .max_voltage = 12.0,
                                    .trackWidthMeters = TRACK_WIDTH_INCHES * lemlib::INCH_TO_METER,
                                    .enableTCS = true,
                                    .maxSlipRatio = 0.28}; // relaxed to prevent false throttle cuts

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

// ============================================================================
// 4. Unified Autonomous Motion Engine (LTV + LQR + PID)
// ============================================================================

/**
 * @brief 1. Smooth Jerk-Limited Quintic Hermite Spline Trajectory (LTV-LQR + TCS)
 * Computes optimal C^2 continuous trajectory on-the-fly and tracks with online DARE Riccati solver.
 */
void autoSpline(lemlib::Pose start, lemlib::Pose end, double maxVel = 1.0, double maxAccel = 1.8, double maxJerk = 3.5) {
    lemlib::QuinticSplineGenerator::SplineWaypoints params {
        .start = start,
        .end = end,
        .startVel = 0.0,
        .endVel = 0.0,
        .maxVel = maxVel,
        .maxAccel = maxAccel,
        .maxJerk = maxJerk
    };
    auto traj = lemlib::QuinticSplineGenerator::generateTrajectory(params, 0.01);
    ltvFollower.followTrajectory(traj, {.log = true});
    ltvFollower.waitUntilDone();
}

/**
 * @brief 2. Multi-Waypoint Continuous S-Curve Trajectory (LTV-LQR)
 */
void autoPath(const std::vector<lemlib::Pose>& waypoints, double maxVel = 1.0, double maxAccel = 1.8) {
    auto traj = lemlib::QuinticSplineGenerator::generateMultiPointTrajectory(waypoints, maxVel, maxAccel, 0.01);
    ltvFollower.followTrajectory(traj, {.log = true});
    ltvFollower.waitUntilDone();
}

/**
 * @brief 3. Fast Heading Snap Turn (Optimal LQR State Feedback)
 * Delivers 0.5-degree accuracy in <700ms with zero throttling.
 */
void autoTurn(float targetHeading, int timeout = 1200) {
    chassis.useLQR();
    chassis.turnToHeading(targetHeading, timeout);
    chassis.waitUntilDone();
}

/**
 * @brief 4. High-Precision Point Drive (LQR State Feedback)
 */
void autoDrive(float x, float y, int timeout = 2000, float maxSpeed = 127.0f) {
    chassis.useLQR();
    chassis.moveToPoint(x, y, timeout, {.maxSpeed = maxSpeed});
    chassis.waitUntilDone();
}

/**
 * @brief 5. High-Precision Pose Drive with Heading (LQR State Feedback)
 */
void autoPose(float x, float y, float theta, int timeout = 2500, float maxSpeed = 127.0f) {
    chassis.useLQR();
    chassis.moveToPose(x, y, theta, timeout, {.maxSpeed = maxSpeed});
    chassis.waitUntilDone();
}

/**
 * @brief 6. Classic PID Distance Drive (Linear PID)
 * Ideal for short linear drives, goal alignment, or squaring against perimeter.
 */
void autoPIDDrive(float targetInches, int timeout = 1500) {
    chassis.usePID();
    lemlib::Pose cur = chassis.getPose();
    float targetX = cur.x + targetInches * std::sin(lemlib::degToRad(cur.theta));
    float targetY = cur.y + targetInches * std::cos(lemlib::degToRad(cur.theta));
    chassis.moveToPoint(targetX, targetY, timeout);
    chassis.waitUntilDone();
}

/**
 * @brief 7. Classic PID Turn (Angular PID)
 */
void autoPIDTurn(float targetHeading, int timeout = 1500) {
    chassis.usePID();
    chassis.turnToHeading(targetHeading, timeout);
    chassis.waitUntilDone();
}

// Mechanism helpers
inline void setIntake(int voltage_mv) { intakeMotors.move_voltage(voltage_mv); }
inline void stopIntake() { intakeMotors.move_voltage(0); }
inline void intake() { setIntake(12000); }
inline void outtake() { setIntake(-12000); }

// ============================================================================
// 5. Interactive Tuning & Calibration Suite (Preserved for Driver Practice)
// ============================================================================

void testTrackWidth(int fullRotations = 1) {
    std::cout << "\n=== STARTING TRACK WIDTH CALIBRATION (" << (360 * fullRotations) << " deg) ===" << std::endl;
    controller.print(0, 0, "Spinning 360...");
    chassis.setPose(0, 0, 0);

    for (int i = 0; i < fullRotations; i++) {
        chassis.turnToHeading(180.0f, 2500, {.direction = lemlib::AngularDirection::CW_CLOCKWISE, .minSpeed = 40, .earlyExitRange = 15});
        chassis.turnToHeading(0.0f, 2500, {.direction = lemlib::AngularDirection::CW_CLOCKWISE});
        chassis.waitUntilDone();
    }

    lemlib::Pose endPose = chassis.getPose();
    std::cout << "Target Angle:   " << (360.0f * fullRotations) << " deg" << std::endl;
    std::cout << "Measured Angle: " << endPose.theta << " deg" << std::endl;
    controller.print(0, 0, "Done: %5.1f deg  ", endPose.theta);
    controller.rumble(".");
}

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
    std::cout << "Final Y: " << endPose.y << " in | Error: " << error << " in | Settle Time: " << elapsed << " ms" << std::endl;
    controller.print(0, 0, "Err: %4.2fin %4dms", error, (int)elapsed);
    controller.rumble(".");
}

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
    std::cout << "Final Theta: " << endPose.theta << " deg | Error: " << error << " deg | Settle: " << elapsed << " ms" << std::endl;
    controller.print(0, 0, "Err: %4.1fdeg %4dms", error, (int)elapsed);
    controller.rumble(".");
}

void testLTVTrajectory() {
    std::cout << "\n=== STARTING LTV S-CURVE TRAJECTORY TEST ===" << std::endl;
    controller.print(0, 0, "Test: LTV S-Curve...");
    chassis.setPose(0, 0, 0);

    std::vector<lemlib::State> path;
    int steps = 100;
    double dt = 0.02;
    for (int i = 0; i <= steps; ++i) {
        double t = (double)i / steps;
        double x = 0.4 * std::sin(M_PI * t);
        double y = 1.8 * t;
        double v = 0.9 * std::sin(M_PI * t);
        double heading = std::atan2(0.4 * M_PI * std::cos(M_PI * t), 1.8);
        double w = (i > 0) ? (heading - path.back().heading) / dt : 0.0;
        path.push_back({x, y, heading, v, w});
    }

    ltvFollower.followTrajectory(path, {.log = true});
    ltvFollower.waitUntilDone();

    controller.print(0, 0, "LTV Done!          ");
    controller.rumble(".");
}

void testFeedforwardCharacterization() {
    std::cout << "\n=== STARTING FEEDFORWARD CHARACTERIZATION ===" << std::endl;
    controller.print(0, 0, "Calibrating kS...");

    double found_kS = 0.0;
    for (int mv = 100; mv <= 4000; mv += 50) {
        leftMotors.move_voltage(mv);
        rightMotors.move_voltage(mv);
        pros::delay(80);

        double velLeft = std::abs(leftMotors.get_actual_velocity(0));
        double velRight = std::abs(rightMotors.get_actual_velocity(0));

        if (velLeft > 5.0 || velRight > 5.0) {
            found_kS = mv / 1000.0;
            std::cout << ">>> MEASURED kS: " << found_kS << " Volts" << std::endl;
            break;
        }
    }
    leftMotors.brake();
    rightMotors.brake();

    controller.print(0, 0, "kS = %4.2f Volts   ", found_kS);
    controller.rumble("-");
}

void testQuinticSpline() {
    std::cout << "\n=== GENERATING JERK-LIMITED QUINTIC SPLINE TRAJECTORY ===" << std::endl;
    controller.print(0, 0, "Gen Spline 5th...");
    chassis.setPose(0, 0, 0);

    autoSpline(lemlib::Pose(0, 0, 0), lemlib::Pose(20.0, 40.0, 45.0), 1.0, 1.8, 3.5);

    controller.print(0, 0, "Spline Done!       ");
    controller.rumble(".");
}

// ============================================================================
// 6. Autonomous Routine Definitions (AWP, Goal Rush, Skills)
// ============================================================================

enum class AutoRoutine {
    HYBRID_TEST = 0,    // Combined LTV Spline + LQR Snap + PID Wall align
    RED_SOLO_AWP = 1,   // Red Alliance Solo Win Point
    BLUE_SOLO_AWP = 2,  // Blue Alliance Solo Win Point
    RED_GOAL_RUSH = 3,  // Fast Center Mobile Goal Rush
    BLUE_GOAL_RUSH = 4, // Fast Center Mobile Goal Rush
    SKILLS_60S = 5      // 60-Second Full Field Skills Autonomous
};

// Default competition routine
AutoRoutine currentAuto = AutoRoutine::HYBRID_TEST;

/**
 * @brief Hybrid Autonomous Demo (LTV Trajectory + LQR Snap Turns + PID Fine Alignment)
 */
void autoHybridDemo() {
    chassis.setPose(0, 0, 0);
    controller.print(0, 0, "Auto: Hybrid Demo  ");

    // Phase 1: LTV-LQR Smooth Curved Path with online DARE Riccati solver
    std::cout << "[Auto] Phase 1: LTV Quintic Spline..." << std::endl;
    autoSpline(lemlib::Pose(0, 0, 0), lemlib::Pose(18.0, 36.0, 45.0), 1.0, 1.8, 3.5);

    // Phase 2: High-Speed LQR Snap Turn to 90 degrees
    std::cout << "[Auto] Phase 2: LQR Snap Turn (90 deg)..." << std::endl;
    autoTurn(90.0f, 1000);

    // Phase 3: LQR Point-to-Point Precision Drive
    std::cout << "[Auto] Phase 3: LQR Drive to (30, 36)..." << std::endl;
    autoDrive(30.0f, 36.0f, 1500);

    // Phase 4: LQR Snap Turn back to 0
    std::cout << "[Auto] Phase 4: LQR Snap Turn (0 deg)..." << std::endl;
    autoTurn(0.0f, 1000);

    // Phase 5: Classic PID Micro-alignment
    std::cout << "[Auto] Phase 5: PID fine alignment (-6 in)..." << std::endl;
    autoPIDDrive(-6.0f, 1000);

    controller.print(0, 0, "Auto Complete!     ");
    controller.rumble("..");
}

/**
 * @brief Red Alliance Solo AWP Autonomous Routine
 */
void autoRedAWP() {
    chassis.setPose(0, 0, 0);
    controller.print(0, 0, "Auto: Red Solo AWP ");

    // 1. Score Alliance Stake using LQR Drive
    intake();
    autoDrive(0.0f, 14.0f, 1200);
    pros::delay(300);

    // 2. LTV S-Curve to Mobile Goal
    autoSpline(chassis.getPose(), lemlib::Pose(-16.0, 32.0, -45.0), 1.1, 1.8);

    // 3. LQR Snap Turn to clamp Mobile Goal
    autoTurn(-135.0f, 900);
    autoDrive(-24.0f, 24.0f, 1200);

    // 4. LTV Spline through Ring Stack
    autoSpline(chassis.getPose(), lemlib::Pose(-36.0, 48.0, 0.0), 1.0, 1.8);

    stopIntake();
    controller.print(0, 0, "Red AWP Complete!  ");
}

/**
 * @brief Blue Alliance Solo AWP Autonomous Routine (Mirrored)
 */
void autoBlueAWP() {
    chassis.setPose(0, 0, 0);
    controller.print(0, 0, "Auto: Blue Solo AWP");

    // 1. Score Alliance Stake
    intake();
    autoDrive(0.0f, 14.0f, 1200);
    pros::delay(300);

    // 2. LTV S-Curve to Mobile Goal
    autoSpline(chassis.getPose(), lemlib::Pose(16.0, 32.0, 45.0), 1.1, 1.8);

    // 3. LQR Snap Turn to clamp Mobile Goal
    autoTurn(135.0f, 900);
    autoDrive(24.0f, 24.0f, 1200);

    // 4. LTV Spline through Ring Stack
    autoSpline(chassis.getPose(), lemlib::Pose(36.0, 48.0, 0.0), 1.0, 1.8);

    stopIntake();
    controller.print(0, 0, "Blue AWP Complete! ");
}

/**
 * @brief Center Mobile Goal Rush Autonomous Routine
 */
void autoGoalRush(bool isRedAlliance) {
    chassis.setPose(0, 0, 0);
    float mirror = isRedAlliance ? 1.0f : -1.0f;
    controller.print(0, 0, "Auto: Goal Rush    ");

    // Explosive 1.1 m/s LTV spline rush to center goal
    intake();
    autoSpline(lemlib::Pose(0, 0, 0), lemlib::Pose(mirror * 12.0f, 48.0f, mirror * 15.0f), 1.1, 2.0);

    // LQR Instant Snap Turn & pull back
    autoTurn(mirror * 180.0f, 800);
    autoDrive(mirror * 12.0f, 12.0f, 1500);

    stopIntake();
}

/**
 * @brief 60-Second Full Field Skills Autonomous Routine
 */
void autoSkills() {
    chassis.setPose(0, 0, 0);
    controller.print(0, 0, "Auto: Skills 60s   ");

    // 1. Score Alliance Stake
    intake();
    pros::delay(400);

    // 2. LTV S-Curve to Quadrant 1 Mobile Goal
    autoSpline(lemlib::Pose(0, 0, 0), lemlib::Pose(-18.0, 24.0, -90.0), 1.1, 1.8);

    // 3. LQR Snap Turn & Clamp
    autoTurn(-180.0f, 800);
    autoDrive(-18.0f, 12.0f, 1000);

    // 4. LTV Multi-Waypoint Ring Sweep
    autoPath({
        lemlib::Pose(-18.0, 12.0, -180.0),
        lemlib::Pose(-48.0, 24.0, -90.0),
        lemlib::Pose(-48.0, 48.0, 0.0),
        lemlib::Pose(-24.0, 60.0, 45.0)
    }, 1.0, 1.6);

    // 5. LQR Corner Placement
    autoTurn(135.0f, 900);
    autoDrive(-60.0f, 60.0f, 1500);

    stopIntake();
    controller.print(0, 0, "Skills Done!       ");
}

// ============================================================================
// 7. Lifecycle Functions & Autonomous Selector
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
            robotEKF.updatePose(rawPose.x * lemlib::INCH_TO_METER, rawPose.y * lemlib::INCH_TO_METER, rawPose.theta);

            // 3. Telemetry on Brain LCD
            lemlib::Pose fusedPose = robotEKF.getPose();
            pros::lcd::print(0, "EKF X: %5.1f in | Odom: %5.1f", fusedPose.x, rawPose.x);
            pros::lcd::print(1, "EKF Y: %5.1f in | Odom: %5.1f", fusedPose.y, rawPose.y);
            pros::lcd::print(2, "EKF Th: %5.1f deg", fusedPose.theta);
            pros::lcd::print(3, "Mode: LTV+LQR+PID [ACTIVE]");
            pros::delay(10);
        }
    });
}

void disabled() {}

void competition_initialize() {}

// ============================================================================
// 8. Autonomous Routine Dispatcher
// ============================================================================

void autonomous() {
    switch (currentAuto) {
        case AutoRoutine::HYBRID_TEST:
            autoHybridDemo();
            break;
        case AutoRoutine::RED_SOLO_AWP:
            autoRedAWP();
            break;
        case AutoRoutine::BLUE_SOLO_AWP:
            autoBlueAWP();
            break;
        case AutoRoutine::RED_GOAL_RUSH:
            autoGoalRush(true);
            break;
        case AutoRoutine::BLUE_GOAL_RUSH:
            autoGoalRush(false);
            break;
        case AutoRoutine::SKILLS_60S:
            autoSkills();
            break;
    }
}

// ============================================================================
// 9. Interactive Driver Control & Live Tuning Suite
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

        // --- BUTTON L1: Run Full Hybrid Autonomous Routine (Live in Driver Mode) ---
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_L1)) { autoHybridDemo(); }

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
