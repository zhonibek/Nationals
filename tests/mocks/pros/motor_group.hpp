#pragma once
#include <vector>
#include "pros/motors.h"
namespace pros{class MotorGroup{
 std::vector<int> ports;
public:
 double testRpm=0;
 MotorGroup(std::initializer_list<int> p):ports(p){}unsigned size(){return ports.size();}
 std::vector<int> get_port_all(){return ports;}
 std::vector<double> get_actual_velocity_all(){return std::vector<double>(ports.size(),testRpm);}
 int move(int v){for(int p:ports)c::motor_move(p,v);return 1;}
 int move_voltage(int v){for(int p:ports)if(c::motor_move_voltage(p,v)==PROS_ERR)return PROS_ERR;return 1;}
 void brake(){move(0);}
};}
