#include "subsystems/control/Cascade.hpp"
#include "subsystems/trajectory/QuinticSpline.hpp"
#include "subsystems/control/BezierReference.hpp"
#include <memory>
static std::vector<lemlib::State> trajectory;
static std::unique_ptr<nationals::BezierReference> bezier;
static nationals::Cascade controller;
static double buffer[64];
extern "C" {
void control_curve_eval(double t){
    pedro::BezierCurve c({float(buffer[0]),float(buffer[1])},{float(buffer[2]),float(buffer[3])},{float(buffer[4]),float(buffer[5])},{float(buffer[6]),float(buffer[7])});
    const auto p=c.getPoint(t),d=c.getDerivative(t),dd=c.getSecondDerivative(t);
    buffer[32]=p.x;buffer[33]=p.y;buffer[34]=d.x;buffer[35]=d.y;buffer[36]=dd.x;buffer[37]=dd.y;buffer[38]=c.getCurvature(t);
}
double control_bezier(double x0,double y0,double x1,double y1,double x2,double y2,double x3,double y3,double h0,double h1){
    pedro::BezierCurve c({float(x0),float(y0)},{float(x1),float(y1)},{float(x2),float(y2)},{float(x3),float(y3)});
    bezier=std::make_unique<nationals::BezierReference>(c,h0,h1);return bezier->seconds;
}
int control_bezier_reference(double t){
    if(!bezier)return 0;const auto r=bezier->at(t);
    buffer[0]=r.pose.x;buffer[1]=r.pose.y;buffer[2]=r.pose.h;buffer[3]=r.vx;buffer[4]=r.vy;buffer[5]=r.w;return 1;
}
int control_spline(double ax,double ay,double ah,double bx,double by,double bh,double speed,double accel,double jerk,double dt){
    lemlib::QuinticSplineGenerator::SplineWaypoints p;
    p.start={float(ax),float(ay),float(ah)};p.end={float(bx),float(by),float(bh)};
    p.maxVel=speed;p.maxAccel=accel;p.maxJerk=jerk;
    trajectory=lemlib::QuinticSplineGenerator::generateTrajectory(p,dt);
    return trajectory.size();
}
int control_spline_sample(int index){
    if(index<0||index>=int(trajectory.size()))return 0;
    const auto& s=trajectory[index];buffer[0]=s.x;buffer[1]=s.y;buffer[2]=s.heading;buffer[3]=s.linear_vel;buffer[4]=s.angular_vel;buffer[5]=s.time;return 1;
}
double* control_buffer(){return buffer;}
void control_reset(){controller.reset();}
void control_config(double radius,double maxSpeed,double kV){controller.config.radius=radius;controller.config.maxWheelSpeed=maxSpeed;controller.config.kV=kV;controller.reset();}
int control_step(){
    nationals::Reference r{{buffer[0],buffer[1],buffer[2]},buffer[3],buffer[4],buffer[5]};
    nationals::Feedback f{{buffer[6],buffer[7],buffer[8]},buffer[9],buffer[10],buffer[11],{buffer[12],buffer[13],buffer[14],buffer[15]},buffer[17]>0};
    auto o=controller.update(r,f,buffer[16]);
    for(int i=0;i<4;++i){buffer[32+i]=o.volts[i];buffer[36+i]=o.target[i];}
    return o.valid;
}
double control_duration(double ax,double ay,double ah,double bx,double by,double bh,double speed,double accel,double jerk){return nationals::duration({ax,ay,ah},{bx,by,bh},speed,accel,jerk);}
void control_reference(double ax,double ay,double ah,double bx,double by,double bh,double time,double duration){
    auto r=nationals::pointReference({ax,ay,ah},{bx,by,bh},time,duration);
    buffer[0]=r.pose.x;buffer[1]=r.pose.y;buffer[2]=r.pose.h;buffer[3]=r.vx;buffer[4]=r.vy;buffer[5]=r.w;
}
}
