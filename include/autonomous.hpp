#pragma once
#include "lemlib/api.hpp"
#include <vector>
// Competition motion API: every function uses the same LTV-LQR -> wheel PID
// cascade. Distances are inches; headings clockwise from north, degrees.
// timeout=0 derives a timeout from the feasible reference duration.
lemlib::MotionResult autoPose(float x,float y,float heading,int timeout=0,float maxSpeed=127);
lemlib::MotionResult autoDrive(float x,float y,int timeout=0,float maxSpeed=127);
lemlib::MotionResult autoTurn(float heading,int timeout=0);
lemlib::MotionResult drive(float inches,float heading,int timeout=0,float maxSpeed=127);
lemlib::MotionResult curve(float forward,float right,float heading,int timeout=0,float maxSpeed=127);
lemlib::MotionResult holonomicStrafe(float inches,float heading,int timeout=0,float maxSpeed=127);
lemlib::MotionResult holonomicDiagonal(float forward,float right,float heading,int timeout=0,float maxSpeed=127);
// Legacy names retained for source compatibility; they no longer switch modes.
lemlib::MotionResult autoPIDDrive(float inches,int timeout=0);
lemlib::MotionResult autoPIDTurn(float heading,int timeout=0);
lemlib::MotionResult autoSpline(float x,float y,float heading);
lemlib::MotionResult autoSpline(lemlib::Pose start,lemlib::Pose end,double maxVel=.45,double maxAccel=.8,double maxJerk=3.5);
lemlib::MotionResult autoPath(const std::vector<lemlib::Pose>& waypoints,double maxVel=.45,double maxAccel=.8);
lemlib::MotionResult splineCurve(float forward,float right,float heading,double maxVel=.45,double maxAccel=.8);
