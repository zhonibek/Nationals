#include "main.h"
#include "lemlib/api.hpp" // IWYU pragma: keep

// controller
pros::Controller controller(pros::E_CONTROLLER_MASTER);

// motor groups (6-motor drive, 200 RPM green cartridges)
// Left side: Port 1 (forward), Port 9 (forward), Port 20 (reversed)
pros::MotorGroup leftMotors({1, 9, -20}, pros::MotorGearset::green);

// Right side: Port 11 (reversed), Port 17 (reversed), Port 10 (forward)
pros::MotorGroup rightMotors({-11, -17, 10}, pros::MotorGearset::green);

// drivetrain settings
lemlib::Drivetrain drivetrain(&leftMotors, // left motor group
                              &rightMotors, // right motor group
                              10.5, // 10.5 inch track width
                              lemlib::Omniwheel::NEW_4, // wheel diameter (adjust if using 3.25" or 2.75")
                              200, // drivetrain rpm is 200 (ratio 18:1 green cartridge)
                              2 // horizontal drift
);

// lateral motion controller (PID)
lemlib::ControllerSettings linearController(10, // proportional gain (kP)
                                            0, // integral gain (kI)
                                            3, // derivative gain (kD)
                                            3, // anti windup
                                            1, // small error range, in inches
                                            100, // small error range timeout, in milliseconds
                                            3, // large error range, in inches
                                            500, // large error range timeout, in milliseconds
                                            20 // maximum acceleration (slew)
);

// angular motion controller (PID)
lemlib::ControllerSettings angularController(2, // proportional gain (kP)
                                             0, // integral gain (kI)
                                             10, // derivative gain (kD)
                                             3, // anti windup
                                             1, // small error range, in degrees
                                             100, // small error range timeout, in milliseconds
                                             3, // large error range, in degrees
                                             500, // large error range timeout, in milliseconds
                                             0 // maximum acceleration (slew)
);

// lateral motion controller (LQR with velocity state feedback from motor encoders)
lemlib::LQRSettings lateralLQR(10, // position error gain (kP)
                               3.2, // velocity damping gain (kV)
                               0, // predictive acceleration gain (kA)
                               0, // integral gain (kI)
                               3, // anti windup
                               1, // small error range, in inches
                               100, // small error range timeout, in milliseconds
                               3, // large error range, in inches
                               500, // large error range timeout, in milliseconds
                               20, // maximum acceleration (slew)
                               false // no IMU acceleration sensor present
);

// angular motion controller (LQR with differential velocity damping)
lemlib::LQRSettings angularLQR(2.2, // angular error gain (kP)
                               0.85, // angular velocity damping gain (kV)
                               0, // predictive angular accel gain (kA)
                               0, // integral gain (kI)
                               3, // anti windup
                               1, // small error range, in degrees
                               100, // small error range timeout, in milliseconds
                               3, // large error range, in degrees
                               500, // large error range timeout, in milliseconds
                               0, // maximum acceleration (slew)
                               false // no IMU gyro present
);

// sensors for odometry
// All set to nullptr: LemLib will automatically use the left and right motor encoders for odometry tracking
lemlib::OdomSensors sensors(nullptr, // vertical tracking wheel 1 (nullptr -> uses leftMotors)
                            nullptr, // vertical tracking wheel 2 (nullptr -> uses rightMotors)
                            nullptr, // horizontal tracking wheel 1
                            nullptr, // horizontal tracking wheel 2
                            nullptr // inertial sensor (none)
);

// input curve for throttle input during driver control
lemlib::ExpoDriveCurve throttleCurve(3, // joystick deadband out of 127
                                     10, // minimum output where drivetrain will move out of 127
                                     1.019 // expo curve gain
);

// input curve for steer input during driver control
lemlib::ExpoDriveCurve steerCurve(3, // joystick deadband out of 127
                                  10, // minimum output where drivetrain will move out of 127
                                  1.019 // expo curve gain
);

// create the chassis with both PID and LQR configurations
lemlib::Chassis chassis(drivetrain, linearController, angularController, lateralLQR, angularLQR, sensors,
                        &throttleCurve, &steerCurve);

/**
 * Runs initialization code. This occurs as soon as the program is started.
 *
 * All other competition modes are blocked by initialize; it is recommended
 * to keep execution time for this mode under a few seconds.
 */
void initialize() {
    pros::lcd::initialize(); // initialize brain screen
    chassis.calibrate(); // calibrate sensors

    // Default controller is PID, but you can switch to LQR by uncommenting:
    // chassis.useLQR(); // enable LQR with predictive IMU state feedback
    // or:
    // chassis.setMotionControllerType(lemlib::MotionControllerType::LQR);

    // thread for brain screen and position logging
    pros::Task screenTask([&]() {
        while (true) {
            // print robot location and active controller to the brain screen
            pros::lcd::print(0, "X: %f", chassis.getPose().x); // x
            pros::lcd::print(1, "Y: %f", chassis.getPose().y); // y
            pros::lcd::print(2, "Theta: %f", chassis.getPose().theta); // heading
            pros::lcd::print(3, "Mode: %s",
                             chassis.getMotionControllerType() == lemlib::MotionControllerType::LQR ? "LQR" : "PID");
            // log position telemetry
            lemlib::telemetrySink()->info("Chassis pose: {}", chassis.getPose());
            // delay to save resources
            pros::delay(50);
        }
    });
}

/**
 * Runs while the robot is disabled
 */
void disabled() {}

/**
 * runs after initialize if the robot is connected to field control
 */
void competition_initialize() {}

// get a path used for pure pursuit
// this needs to be put outside a function
ASSET(example_txt); // '.' replaced with "_" to make c++ happy

/**
 * Runs during auto
 *
 * This example demonstrates switching between PID and LQR controllers in autonomous
 */
void autonomous() {
    // --- Example 1: Run movement using LQR (State Feedback + IMU predictive lookahead) ---
    chassis.useLQR(); // switch to LQR mode
    // Move to x: 20 and y: 15, and face heading 90 using LQR
    // chassis.moveToPose(20, 15, 90, 4000);
    chassis.turnToHeading(90, 1000);
    chassis.waitUntilDone();

    // --- Example 2: Switch to traditional PID on the fly ---
    // chassis.usePID(); // switch to PID mode
    // // Move to x: 0 and y: 0 and face heading 270, going backwards using PID
    // chassis.moveToPose(0, 0, 270, 4000, {.forwards = false});
    // chassis.waitUntil(10);
    // chassis.cancelMotion();

    // // Turn to face the point x:45, y:-45 with PID
    // chassis.turnToPoint(45, -45, 1000, {.maxSpeed = 60});
    // chassis.waitUntilDone();

    // --- Example 3: Switch back to LQR for fast, smooth turning and path tracking ---
    // chassis.useLQR();
    // chassis.turnToHeading(90, 1000, {.direction = AngularDirection::CW_CLOCKWISE, .minSpeed = 100});
    // chassis.waitUntilDone();

    // Follow pure pursuit path
    chassis.follow(example_txt, 15, 4000, false);
    chassis.waitUntilDone();
}

/**
 * Runs in driver control
 */
void opcontrol() {
    // controller loop
    while (true) {
        // Switcher shortcut: Press 'X' on controller to toggle between PID and LQR motion control
        if (controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_X)) {
            if (chassis.getMotionControllerType() == lemlib::MotionControllerType::PID) {
                chassis.useLQR();
                controller.print(0, 0, "Mode: LQR (Optimal) ");
                controller.rumble(".");
            } else {
                chassis.usePID();
                controller.print(0, 0, "Mode: PID (Standard)");
                controller.rumble("-");
            }
        }

        // get joystick positions
        int leftY = controller.get_analog(pros::E_CONTROLLER_ANALOG_LEFT_Y);
        int rightX = controller.get_analog(pros::E_CONTROLLER_ANALOG_RIGHT_X);
        // move the chassis with arcade drive
        chassis.arcade(leftY, rightX);
        // delay to save resources
        pros::delay(10);
    }
}
