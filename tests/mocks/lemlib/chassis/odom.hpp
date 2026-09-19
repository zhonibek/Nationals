#pragma once
#include <cstdint>
namespace lemlib{
struct OdomStatus{bool valid;uint32_t epoch;};
struct TestPose{double x=0,y=0,theta=0;};
struct OdomSnapshot{TestPose pose,speed;OdomStatus status;};
inline bool testOdomValid=true;
inline uint32_t testEpoch=0;
inline TestPose testPose{},testSpeed{};
inline OdomStatus getOdomStatus(){return {testOdomValid,testEpoch};}
inline OdomSnapshot getOdomSnapshot(){return {testPose,testSpeed,getOdomStatus()};}
}
