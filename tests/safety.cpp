// Runs the real safety.cpp against a deterministic hardware boundary, not a
// second implementation. Scheduling/PROS hardware calls still require bench QA.
#include "lemlib/safety.hpp"
#include "lemlib/chassis/odom.hpp"
#include "pros/misc.hpp"
#include "pros/motors.h"
#define CHECK(x) if(!(x))return __LINE__
extern "C" int safety_test(){
 pros::MotorGroup left{-11,-20},right{1,10};lemlib::DriveOutput out;
 CHECK(out.configure(&left,&right));auto token=out.acquire();CHECK(token);CHECK(!out.acquire());
 double v[4]={3,4,5,6};CHECK(out.wheelVoltages(token,v));CHECK(pros::c::voltages[11]==-3000&&pros::c::voltages[1]==5000);
 CHECK(!out.wheelVoltages(token+1,v));CHECK(pros::c::voltages[1]==5000);
 out.release(token+1);CHECK(out.owns(token));out.stop();CHECK(!out.wheelVoltages(token,v));CHECK(pros::c::voltages[1]==0);
 token=out.acquire();CHECK(token);v[2]=NAN;CHECK(!out.wheelVoltages(token,v));CHECK(pros::c::voltages[11]==0&&!out.owns(token));v[2]=5;
 token=out.acquire();CHECK(out.wheelVoltages(token,v));pros::competition::testMode=1;out.watchdog();CHECK(pros::c::voltages[1]==0);CHECK(!out.acquire());pros::competition::testMode=0;
 token=out.acquire();CHECK(out.wheelVoltages(token,v));pros::testTime+=101;out.watchdog();CHECK(!out.owns(token));CHECK(pros::c::voltages[1]==0);
 token=out.acquire();CHECK(out.wheelVoltages(token,v));lemlib::testOdomValid=false;CHECK(!out.wheelVoltages(token,v));CHECK(pros::c::voltages[1]==0);CHECK(!out.acquire());
 token=out.acquire(false);CHECK(token);CHECK(out.wheelVoltages(token,v));out.release(token);lemlib::testOdomValid=true;
 token=out.acquire();CHECK(out.wheelVoltages(token,v));pros::competition::testMode=2;CHECK(!out.owns(token));CHECK(pros::c::voltages[1]==0);pros::competition::testMode=0;
 token=out.acquire();CHECK(token);pros::c::failPort=20;CHECK(!out.wheelVoltages(token,v));CHECK(!out.owns(token));CHECK(pros::c::voltages[1]==0);pros::c::failPort=0;
 return 0;
}
