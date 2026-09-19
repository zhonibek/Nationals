#pragma once
#include "pros/rtos.hpp"
namespace pros::c{
inline int voltages[22]={};inline int failPort=0;
inline int motor_move_voltage(int p,int v){int port=p<0?-p:p;if(port==failPort)return PROS_ERR;voltages[port]=p<0?-v:v;return 1;}
inline int motor_move(int p,int v){return motor_move_voltage(p,v*12000/127);}
}
