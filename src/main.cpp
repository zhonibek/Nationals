#include "main.h"
#include "lemlib/api.hpp"
#include "lemlib/chassis/odom.hpp"
#include "lemlib/safety.hpp"
#include "subsystems/pedro/PedroFollower.hpp"
#include "subsystems/MotorMonitor.hpp"
#include "robot_config.hpp"
#include <atomic>
#include <cmath>

// North (+Y), clockwise headings; distances inches; FL,BL,FR,BR order.
// X-drive effective translation diameter and turning span assume 45-degree wheels.
using namespace robot_config;
pros::Controller controller(pros::E_CONTROLLER_MASTER);
pros::MotorGroup leftMotors({frontLeft,backLeft},pros::MotorGearset::green);
pros::MotorGroup rightMotors({frontRight,backRight},pros::MotorGearset::green);
lemlib::Drivetrain drivetrain(&leftMotors,&rightMotors,widthIn+wheelbaseIn,
    wheelDiameterIn*1.41421356237f,cartridgeRpm*externalGearRatio,2);
lemlib::OdomSensors sensors(nullptr,nullptr,nullptr,nullptr,nullptr);
lemlib::ControllerSettings linear(linear_kP,linear_kI,linear_kD,2,0.35,100,0.9,250,15);
lemlib::ControllerSettings angular(angular_kP,angular_kI,angular_kD,2,0.4,100,1,250,15);
lemlib::LQRSettings linearState(linear_kP,linear_kV,0,linear_kI,2,0.35,100,0.9,250,15,false);
lemlib::LQRSettings angularState(angular_kP,angular_kV,0,angular_kI,2,0.4,100,1,250,15,false);
lemlib::Chassis chassis(drivetrain,linear,angular,linearState,angularState,sensors);
lemlib::MotorMonitor monitor(controller,{{"LeftDrive",&leftMotors},{"RightDrive",&rightMotors}});
pedro::PedroFollower pedroFollower;
std::atomic<int> requestedTest{0};
std::atomic<bool> diagnosticBusy{false}, abortDiagnostic{false};

void stopEverything() {
    abortDiagnostic=true; requestedTest=0;
    pedroFollower.cancel(); lemlib::driveOutput().stop(); chassis.cancelAllMotions();
}
bool diagnosticAllowed(uint32_t lease,int mode) {
    return !abortDiagnostic && mode==pros::competition::get_status() && lemlib::driveOutput().owns(lease);
}
void timedWheelTest(uint32_t lease,int wheel,int mode) {
    const auto start=pros::millis();
    while(pros::millis()-start<1200 && diagnosticAllowed(lease,mode)) {
        float cmd[4]={0,0,0,0}; cmd[wheel]=40;
        if(!lemlib::driveOutput().wheels(lease,cmd[0],cmd[1],cmd[2],cmd[3])) break;
        pros::delay(10);
    }
    lemlib::driveOutput().wheels(lease,0,0,0,0);
}
lemlib::MotionResult strafeTo(float x,float y,float heading) {
    const auto lease=lemlib::driveOutput().acquire();
    if(!lease) return lemlib::MotionResult::Busy;
    auto result=lemlib::MotionResult::TimedOut;
    const int mode=pros::competition::get_status();
    const auto start=pros::millis(); uint32_t settle=0; bool settling=false;
    while(pros::millis()-start<4000 && diagnosticAllowed(lease,mode)) {
        auto snap=lemlib::getOdomSnapshot();
        if(!snap.status.valid) { result=lemlib::MotionResult::SensorFault; break; }
        const auto p=snap.pose;
        const float dx=x-p.x,dy=y-p.y,err=std::hypot(dx,dy);
        const float he=std::remainder(heading-lemlib::radToDeg(p.theta),360.f);
        if(err<0.8f&&std::abs(he)<2&&std::hypot(snap.localSpeed.x,snap.localSpeed.y)<1) {
            if(!settling) {settle=pros::millis();settling=true;}
            if(pros::millis()-settle>=150) {result=lemlib::MotionResult::Settled;break;}
        } else settling=false;
        const float f=7.5f*(dx*std::sin(p.theta)+dy*std::cos(p.theta))-0.42f*snap.localSpeed.y;
        const float s=7.5f*(dx*std::cos(p.theta)-dy*std::sin(p.theta))-0.42f*snap.localSpeed.x;
        const float t=2.8f*he-0.18f*lemlib::radToDeg(snap.localSpeed.theta);
        if(!lemlib::driveOutput().holonomic(lease,f,s,t,80)) {result=lemlib::MotionResult::SensorFault;break;}
        pros::delay(10);
    }
    if(abortDiagnostic||mode!=pros::competition::get_status()) result=lemlib::MotionResult::Cancelled;
    lemlib::driveOutput().release(lease); return result;
}
void runDiagnostic(int test) {
    if(abortDiagnostic) return;
    chassis.setPose(0,0,0);
    auto result=lemlib::MotionResult::Idle;
    const int mode=pros::competition::get_status();
    if(test==1) {
        const auto lease=lemlib::driveOutput().acquire(false);
        if(!lease) return;
        for(int wheel=0;wheel<4&&diagnosticAllowed(lease,mode);++wheel) timedWheelTest(lease,wheel,mode);
        lemlib::driveOutput().release(lease);
        result=abortDiagnostic?lemlib::MotionResult::Cancelled:lemlib::MotionResult::Settled;
    } else if(test==2 || test==3 || test==5) {
        if(test==2) chassis.moveToPoint(0,24,4000,{.maxSpeed=80},false);
        else chassis.turnToHeading(test==3?90:180,4000,{.maxSpeed=70},false);
        result=chassis.getMotionResult();
    } else if(test==4 || test==6) result=strafeTo(24,test==6?24:0,0);
    else if(test==7) {
        const auto lease=lemlib::driveOutput().acquire();
        if(!lease) return;
        pedro::BezierCurve curve({0,0},{0,16},{24,8},{24,24});
        pedro::PedroPath path(curve); path.setLinearHeading(0,90);
        result=pedroFollower.follow(path,6000,[lease](int f,int s,int t) {lemlib::driveOutput().holonomic(lease,f,s,t,80);},
            [lease]{lemlib::driveOutput().release(lease);});
    }
    controller.print(0,0,"Test %d result %d  ",test,static_cast<int>(result));
}
void initialize() {
    pros::lcd::initialize();
    lemlib::driveOutput().configure(&leftMotors,&rightMotors);
    chassis.setDrivebaseType(lemlib::DrivebaseType::XDRIVE);
    chassis.calibrate(); chassis.setPose(0,0,0);
    chassis.setBrakeMode(pros::E_MOTOR_BRAKE_BRAKE);
    monitor.startTask(500);
    static pros::Task diagnostics([] {
        while(true) {
            const int test=requestedTest.exchange(0);
            if(test && !pros::competition::is_disabled() && !pros::competition::is_autonomous()) runDiagnostic(test);
            if(test) diagnosticBusy=false;
            pros::delay(10);
        }
    });
    static pros::Task telemetry([] {
        while(true) {
            auto s=lemlib::getOdomSnapshot();
            pros::lcd::print(0,"X %.2f Y %.2f H %.1f",s.pose.x,s.pose.y,lemlib::radToDeg(s.pose.theta));
            pros::lcd::print(1,"Odom %s fault %d",s.status.valid?"OK":"FAULT",static_cast<int>(s.status.fault));
            pros::delay(100);
        }
    });
}
void disabled() { stopEverything(); }
void competition_initialize() { stopEverything(); }
void autonomous() {
    stopEverything(); abortDiagnostic=false;
    chassis.setPose(0,0,0);
    chassis.moveToPoint(0,24,4000,{.maxSpeed=80},false);
    lemlib::driveOutput().stop();
}
void opcontrol() {
    stopEverything();
    uint32_t lease=0;
    const auto mode=pros::competition::get_status();
    while(mode==pros::competition::get_status() && !pros::competition::is_disabled()) {
        if(controller.get_digital(pros::E_CONTROLLER_DIGITAL_A)) { stopEverything(); lease=0; }
        else if(!diagnosticBusy) {
            int test=0;
            if(controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_UP)) test=1;
            if(controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_B)) test=2;
            if(controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_Y)) test=3;
            if(controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_X)) test=4;
            if(controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_LEFT)) test=5;
            if(controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_DOWN)) test=6;
            if(controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_RIGHT)) test=7;
            if(controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_L1)) {
                lemlib::driveOutput().release(lease); lease=0; chassis.setPose(0,0,0);
            }
            if(test) {
                lemlib::driveOutput().release(lease); lease=0; abortDiagnostic=false; diagnosticBusy=true; requestedTest=test;
            } else {
                if(!lease) lease=lemlib::driveOutput().acquire(false);
                const float f=controller.get_analog(pros::E_CONTROLLER_ANALOG_LEFT_Y);
                const float s=controller.get_analog(pros::E_CONTROLLER_ANALOG_LEFT_X);
                const float t=controller.get_analog(pros::E_CONTROLLER_ANALOG_RIGHT_X);
                if(!lemlib::driveOutput().holonomic(lease,std::abs(f)>5?f:0,std::abs(s)>5?s:0,std::abs(t)>5?t:0)) lease=0;
            }
        }
        pros::delay(10);
    }
    stopEverything();
}
