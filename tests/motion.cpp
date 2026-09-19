// Exercise the production PROS boundary with deterministic clock/sensor doubles.
#include "subsystems/control/HolonomicMotion.hpp"
#include "lemlib/chassis/odom.hpp"
#include "pros/misc.hpp"
#include "pros/motors.h"
#define CHECK(x) if(!(x))return __LINE__
extern "C" int motion_test(){
 using lemlib::MotionResult;
 pros::MotorGroup left{-11,-20},right{1,10};
 auto& out=lemlib::driveOutput();CHECK(out.configure(&left,&right));
 const auto config=nationals::configuredCascade();
 auto idle=[](double){return nationals::Reference{};};
 auto forward=[](double){return nationals::Reference{{0,1,0},0,0,0};};
 auto never=[](){return false;};
 auto run=[&](const std::function<nationals::Reference(double)>& ref,double seconds=.01,double timeout=.5){
   return nationals::followReference(left,right,.08255,1,config,seconds,timeout,ref,never);
 };
 CHECK(run(idle)==MotionResult::Settled);
 CHECK(pros::c::voltages[11]==0&&pros::c::voltages[1]==0);
 CHECK(run(forward,.01,.05)==MotionResult::TimedOut);
 CHECK(pros::c::voltages[11]==0&&pros::c::voltages[1]==0);
 CHECK(nationals::followReference(left,right,0,1,config,.01,.5,idle,never)==MotionResult::InvalidInput);
 CHECK(nationals::followReference(left,right,.08255,1,config,.01,.5,idle,[]{return true;})==MotionResult::Cancelled);
 auto token=out.acquire();CHECK(token);CHECK(run(idle)==MotionResult::Busy);out.release(token);
 left.testRpm=NAN;CHECK(run(idle)==MotionResult::SensorFault);left.testRpm=0;
 pros::delayHook=[]{++lemlib::testEpoch;};CHECK(run(forward)==MotionResult::SensorFault);pros::delayHook=nullptr;
 pros::delayHook=[]{pros::testTime+=200;};CHECK(run(forward)==MotionResult::SensorFault);pros::delayHook=nullptr;
 pros::delayHook=[]{pros::competition::testMode=1;};CHECK(run(forward)==MotionResult::Cancelled);pros::delayHook=nullptr;pros::competition::testMode=0;
 CHECK(pros::c::voltages[11]==0&&pros::c::voltages[1]==0);
 CHECK(out.configure(nullptr,nullptr)==false);
 return 0;
}
