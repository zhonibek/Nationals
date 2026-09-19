#include "main.h"
#include "lemlib/api.hpp"
#include "lemlib/chassis/odom.hpp"
#include "lemlib/safety.hpp"
#include "subsystems/pedro/PedroFollower.hpp"
#include "subsystems/MotorMonitor.hpp"
#include "subsystems/control/HolonomicMotion.hpp"
#include "subsystems/control/BezierReference.hpp"
#include "subsystems/trajectory/QuinticSpline.hpp"
#include "subsystems/ltv/ltv.hpp"
#include "robot_config.hpp"
#include "autonomous.hpp"
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
VelocityControllerConfig velocityConfiguration() {
    const auto c=nationals::configuredCascade();
    VelocityControllerConfig v;
    v.kV=c.kV; v.KS_straight=c.kS; v.KA_straight=c.kA;
    v.KP_straight=c.wheelKp; v.KI_straight=c.wheelKi;
    v.wheelDiameterMeters=wheelDiameterIn*0.0254;
    v.externalGearRatio=externalGearRatio;
    v.trackWidthMeters=(widthIn+wheelbaseIn)*0.0254;
    return v;
}
lemlib::LTVPathFollower ltvFollower(chassis,leftMotors,rightMotors,velocityConfiguration());
std::atomic<int> requestedTest{0};
std::atomic<bool> diagnosticBusy{false}, abortDiagnostic{false};
std::atomic<int> selectedRoutine{0};
const char* routineNames[]={"Cascade demo","Holonomic skills","Pedro Bezier"};

void stopEverything() {
    abortDiagnostic=true;
    if(requestedTest.exchange(0)!=0) diagnosticBusy=false;
    pedroFollower.cancel(); ltvFollower.cancel(); lemlib::driveOutput().stop(); chassis.cancelAllMotions();
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
// All X-drive point motions use an outer pose loop and four inner wheel loops.
lemlib::MotionResult autoPose(float x,float y,float heading,int timeout,float maxSpeed) {
    if(timeout<0||!std::isfinite(maxSpeed)||maxSpeed<=0||maxSpeed>127)return lemlib::MotionResult::InvalidInput;
    const auto snapshot=lemlib::getOdomSnapshot();
    if(!snapshot.status.valid) return lemlib::MotionResult::SensorFault;
    const nationals::Pose start{snapshot.pose.x*0.0254,snapshot.pose.y*0.0254,snapshot.pose.theta};
    const nationals::Pose end{x*0.0254,y*0.0254,heading*nationals::pi/180};
    const double duration=nationals::duration(start,end,0.45*maxSpeed/127);
    return nationals::followReference(leftMotors,rightMotors,wheelDiameterIn*0.0254,
        externalGearRatio,nationals::configuredCascade(),duration,timeout?timeout*.001:duration+3,
        [=](double time){return nationals::pointReference(start,end,time,duration);},
        []{return abortDiagnostic.load();});
}
lemlib::MotionResult strafeTo(float x,float y,float heading) {return autoPose(x,y,heading);}
lemlib::MotionResult autoSpline(float x,float y,float heading) {
    return autoSpline(chassis.getPose(),{x,y,heading});
}
lemlib::MotionResult autoSpline(lemlib::Pose start,lemlib::Pose end,double maxVel,double maxAccel,double maxJerk) {
    lemlib::QuinticSplineGenerator::SplineWaypoints p;
    p.start=start; p.end=end;
    p.maxVel=maxVel; p.maxAccel=maxAccel; p.maxJerk=maxJerk;
    const auto trajectory=lemlib::QuinticSplineGenerator::generateTrajectory(p,0.01);
    if(trajectory.empty()) return lemlib::MotionResult::InvalidInput;
    if(abortDiagnostic) return lemlib::MotionResult::Cancelled;
    ltvFollower.followTrajectory(trajectory,{.log=false,.settleTimeout=3});
    ltvFollower.waitUntilDone(); return ltvFollower.getResult();
}
lemlib::MotionResult autoPath(const std::vector<lemlib::Pose>& waypoints,double maxVel,double maxAccel){
    const auto path=lemlib::QuinticSplineGenerator::generateMultiPointTrajectory(waypoints,maxVel,maxAccel,.01);
    if(path.empty())return lemlib::MotionResult::InvalidInput;
    if(abortDiagnostic)return lemlib::MotionResult::Cancelled;
    ltvFollower.followTrajectory(path,{.log=false,.settleTimeout=3});ltvFollower.waitUntilDone();return ltvFollower.getResult();
}
lemlib::MotionResult autoDrive(float x,float y,int timeout,float maxSpeed){return autoPose(x,y,chassis.getPose().theta,timeout,maxSpeed);}
lemlib::MotionResult autoTurn(float h,int timeout){const auto p=chassis.getPose();return autoPose(p.x,p.y,h,timeout);}
lemlib::MotionResult drive(float distance,float heading,int timeout,float maxSpeed){
    const auto p=chassis.getPose();const double h=heading*nationals::pi/180;
    return autoPose(p.x+distance*std::sin(h),p.y+distance*std::cos(h),heading,timeout,maxSpeed);
}
lemlib::MotionResult curve(float forward,float right,float heading,int timeout,float maxSpeed){
    const auto p=chassis.getPose();const double h=p.theta*nationals::pi/180;
    return autoPose(p.x+right*std::cos(h)+forward*std::sin(h),p.y-right*std::sin(h)+forward*std::cos(h),heading,timeout,maxSpeed);
}
lemlib::MotionResult holonomicStrafe(float distance,float heading,int timeout,float maxSpeed){
    const auto p=chassis.getPose();const double h=heading*nationals::pi/180;
    return autoPose(p.x+distance*std::cos(h),p.y-distance*std::sin(h),heading,timeout,maxSpeed);
}
lemlib::MotionResult holonomicDiagonal(float forward,float right,float heading,int timeout,float maxSpeed){return curve(forward,right,heading,timeout,maxSpeed);}
lemlib::MotionResult autoPIDDrive(float distance,int timeout){return drive(distance,chassis.getPose().theta,timeout);}
lemlib::MotionResult autoPIDTurn(float heading,int timeout){return autoTurn(heading,timeout);}
lemlib::MotionResult splineCurve(float forward,float right,float heading,double maxVel,double maxAccel){
    const auto p=chassis.getPose();const double h=p.theta*nationals::pi/180;
    return autoSpline(p,{float(p.x+right*std::cos(h)+forward*std::sin(h)),float(p.y-right*std::sin(h)+forward*std::cos(h)),heading},maxVel,maxAccel);
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
        if(test==2) result=autoPose(0,24,0);
        else result=autoPose(0,0,test==3?90:180);
    } else if(test==8) result=autoSpline(24,24,90);
    else if(test==4 || test==6) result=strafeTo(24,test==6?24:0,0);
    else if(test==7) {
        pedro::BezierCurve curve({0,0},{0,16},{24,8},{24,24});
        nationals::BezierReference path(curve,0,nationals::pi/2);
        result=nationals::followReference(leftMotors,rightMotors,wheelDiameterIn*0.0254,
            externalGearRatio,nationals::configuredCascade(),path.seconds,path.seconds+3,
            [&](double t){return path.at(t);},[]{return abortDiagnostic.load();});
    }
    controller.print(0,0,"Test %d result %d  ",test,static_cast<int>(result));
}
void initialize() {
    pros::lcd::initialize();
    pros::lcd::register_btn1_cb([]{if(pros::competition::is_disabled())selectedRoutine=(selectedRoutine+1)%3;});
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
            pros::lcd::print(2,"Auto: %s",routineNames[selectedRoutine]);
            pros::delay(100);
        }
    });
}
void disabled() { stopEverything(); }
void competition_initialize() { stopEverything(); }
void autonomous() {
    stopEverything();
    // Let any diagnostic observe cancellation before clearing its stop flag.
    while(diagnosticBusy || ltvFollower.isRunning()) pros::delay(10);
    abortDiagnostic=false; chassis.setPose(0,0,0);
    if(selectedRoutine==2)runDiagnostic(7);
    else {
        bool proceed=true;
        if(selectedRoutine==1){
            const lemlib::Pose points[]={{0,24,0},{24,24,0},{0,24,0},{0,0,0}};
            for(const auto& p:points)if(autoPose(p.x,p.y,p.theta)!=lemlib::MotionResult::Settled){proceed=false;break;}
        }
        if(proceed&&autoSpline(24,24,90)==lemlib::MotionResult::Settled) autoPose(0,0,0);
    }
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
            if(controller.get_digital_new_press(pros::E_CONTROLLER_DIGITAL_R1)) test=8;
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
