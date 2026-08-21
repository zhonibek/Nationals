#include "main.h"
#include "lemlib/api.hpp" // IWYU pragma: keep

// ==========================================
// 1. Controller & Hardware Configuration
// ==========================================
pros::Controller controller(pros::E_CONTROLLER_MASTER);

// Motor groups (6-motor drive, 200 RPM green cartridges)
// Adjusted for your physical motor wiring and polarity
pros::MotorGroup leftMotors({-3, 18, -5}, pros::MotorGearset::green);
pros::MotorGroup rightMotors({-10, 3, -17}, pros::MotorGearset::green);

// Optional mechanism motors (e.g. intake / color sorting roller)
pros::MotorGroup intakeMotors({2}, pros::MotorGearset::blue);

// Sensors for Odometry & Localization
// (Set to nullptr if not yet mounted; LemLib will use motor encoders as fallback)
pros::Imu imu(21);
// pros::Rotation verticalRotation(11);
// pros::Rotation horizontalRotation(12);
// pros::Optical opticalSensor(13);
// pros::Distance frontDistance(14);
// pros::Distance backDistance(15);
// pros::Distance leftDistance(16);
// pros::Distance rightDistance(17);

// Drivetrain geometry and kinematics
lemlib::Drivetrain drivetrain(&rightMotors, // right motor group mapped to drivetrain left/right
                              &leftMotors,
                              10.5, // 10.5 inch track width
                              lemlib::Omniwheel::NEW_4, // 4-inch wheels
                              200, // 200 RPM
                              2.0  // horizontal drift
);

// ==========================================
// 2. Motion Controller Configurations
// ==========================================

// Lateral PID Controller
lemlib::ControllerSettings linearController(10, // kP
                                            0,  // kI
                                            3,  // kD
                                            3,  // anti windup
                                            1,  // small error range (in)
                                            100,// small error timeout (ms)
                                            3,  // large error range (in)
                                            500,// large error timeout (ms)
                                            20  // slew rate limit
);

// Angular PID Controller
lemlib::ControllerSettings angularController(2,  // kP
                                             0,  // kI
                                             10, // kD
                                             3,  // anti windup
                                             1,  // small error range (deg)
                                             100,// small error timeout (ms)
                                             3,  // large error range (deg)
                                             500,// large error timeout (ms)
                                             0   // slew rate limit
);

// Lateral LQR Controller (Optimal State Feedback)
lemlib::LQRSettings lateralLQR(10.0, // position gain (kP)
                               3.2,  // velocity damping (kV)
                               0.0,  // predictive accel (kA)
                               0.0,  // integral gain (kI)
                               3.0,  // anti windup
                               1.0,  // small error (in)
                               100,  // small error timeout (ms)
                               3.0,  // large error (in)
                               500,  // large error timeout (ms)
                               20.0, // slew limit
                               false // use IMU prediction (set true if IMU mounted)
);

// Angular LQR Controller (Optimal Heading Control)
lemlib::LQRSettings angularLQR(1.0,  // heading gain (kP)
                               1.8,  // angular damping (kV)
                               0.0,  // predictive accel (kA)
                               0.0,  // integral gain (kI)
                               3.0,  // anti windup
                               1.0,  // small error (deg)
                               100,  // small error timeout (ms)
                               3.0,  // large error (deg)
                               500,  // large error timeout (ms)
                               0.0,  // slew limit
                               false // use IMU prediction
);

// Odometry Sensors Struct
lemlib::OdomSensors sensors(nullptr, // vertical tracking wheel 1
                            nullptr, // vertical tracking wheel 2
                            nullptr, // horizontal tracking wheel 1
                            nullptr, // horizontal tracking wheel 2
                            nullptr  // inertial sensor (set &imu when connected)
);

// Drive Curves for driver control
lemlib::ExpoDriveCurve throttleCurve(3, 10, 1.019);
lemlib::ExpoDriveCurve steerCurve(3, 10, 1.019);

// LemLib Chassis Instance
lemlib::Chassis chassis(drivetrain, linearController, angularController, lateralLQR, angularLQR, sensors,
                        &throttleCurve, &steerCurve);

// ==========================================
// 3. Subsystem Integrations: LTV + Monitors
// ==========================================

// PIDf Voltage Controller configuration for LTV feedforward + feedback
VelocityControllerConfig velConfig{
    .kV = 4.5,
    .KA_straight = 0.2,
    .KA_turn = 0.15,
    .KS_straight = 0.4,
    .KS_turn = 0.3,
    .KP_straight = 2.0,
    .KI_straight = 0.0,
    .max_voltage = 12.0,
    .trackWidthMeters = 10.5 * lemlib::INCH_TO_METER
};

// LTV Path Follower (Online DARE Riccati solver + trajectory tracking)
lemlib::LTVPathFollower ltvFollower(chassis, leftMotors, rightMotors, velConfig);

// Motor & Sensor Watchdog
lemlib::MotorMonitor motorMonitor(controller, {
    {"LeftDrive", &leftMotors},
    {"RightDrive", &rightMotors},
    {"Intake", &intakeMotors}
});

// Color Sorter
lemlib::ColorSorter colorSorter(nullptr, &intakeMotors, lemlib::AllianceColor::RED);

// Pure pursuit sample asset
ASSET(example_txt);

// ==========================================
// 4. Lifecycle Functions
// ==========================================

void initialize() {
    pros::lcd::initialize();
    chassis.calibrate();

    // Start background health monitor task
    motorMonitor.startTask(500);

    // Brain LCD telemetry task
    pros::Task screenTask([&]() {
        while (true) {
            pros::lcd::print(0, "X: %6.2f in", chassis.getPose().x);
            pros::lcd::print(1, "Y: %6.2f in", chassis.getPose().y);
            pros::lcd::print(2, "Theta: %6.2f deg", chassis.getPose().theta);
            pros::lcd::print(3, "Mode: %s",
                             chassis.getMotionControllerType() == lemlib::MotionControllerType::LQR ? "LQR (Optimal)" : "PID (Classic)");
            pros::delay(50);
        }
    });
}

void disabled() {}

void competition_initialize() {}

// ==========================================
// 5. Autonomous Routine
// ==========================================

void autonomous() {
    // --- Step 1: LQR for precise turn and point motions ---
    chassis.useLQR();
    chassis.turnToHeading(90, 1000);
    chassis.waitUntilDone();

    // --- Step 2: Drive forward to target using LQR ---
    chassis.moveToPoint(0, 24, 2500);
    chassis.waitUntilDone();

    // --- Step 3: LTV Path Following Example ---
    // (Follows high-speed curved trajectory generated with feedforward velocity profiles)
    // ltvFollower.followPath("path_data", {.turnFirst = true, .log = true});
    // ltvFollower.waitUntilDone();

    // --- Step 4: Pure pursuit path following ---
    // chassis.follow(example_txt, 15, 4000, false);
    // chassis.waitUntilDone();
}

// ==========================================
// 6. Driver Control
// ==========================================

void opcontrol() {
    while (true) {
        // Toggle Motion Controller: Press 'X' to switch between LQR and PID modes
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

        // Toggle Color Sorter Alliance: Press 'Y' to switch RED <-> BLUE <-> DISABLED
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_Y)) {
            if (colorSorter.getAlliance() == lemlib::AllianceColor::RED) {
                colorSorter.setAlliance(lemlib::AllianceColor::BLUE);
                controller.print(0, 0, "Sort: BLUE Alliance");
            } else if (colorSorter.getAlliance() == lemlib::AllianceColor::BLUE) {
                colorSorter.setAlliance(lemlib::AllianceColor::DISABLED);
                controller.print(0, 0, "Sort: DISABLED     ");
            } else {
                colorSorter.setAlliance(lemlib::AllianceColor::RED);
                controller.print(0, 0, "Sort: RED Alliance ");
            }
        }

        // Get joystick positions
        int leftY = controller.get_analog(pros::E_CONTROLLER_ANALOG_LEFT_Y);   // Axis 3 (Throttle)
        int rightX = controller.get_analog(pros::E_CONTROLLER_ANALOG_RIGHT_X); // Axis 1 (Turn)

        // Arcade drive with smooth exponential response curves
        chassis.arcade(leftY, rightX);

        pros::delay(10);
    }
}
