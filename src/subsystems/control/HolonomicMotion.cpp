#include "subsystems/control/HolonomicMotion.hpp"
#include "lemlib/chassis/odom.hpp"
#include "robot_config.hpp"
#include "pros/misc.hpp"
namespace nationals {
Config configuredCascade(){
    Config c;c.radius=(robot_config::widthIn+robot_config::wheelbaseIn)*0.0254/(2*sqrt(2.0));
    c.maxWheelSpeed=robot_config::cartridgeRpm*robot_config::externalGearRatio*pi*robot_config::wheelDiameterIn*0.0254/60;
    c.kV=12/c.maxWheelSpeed;return c;
}
Feedback readFeedback(pros::MotorGroup& left,pros::MotorGroup& right,double diameter,double ratio,uint32_t expectedEpoch){
    const auto o=lemlib::getOdomSnapshot();Feedback f;
    f.pose={o.pose.x*0.0254,o.pose.y*0.0254,o.pose.theta};
    f.vx=o.speed.x*0.0254;f.vy=o.speed.y*0.0254;f.w=o.speed.theta;f.valid=o.status.valid&&o.status.epoch==expectedEpoch;
    const auto l=left.get_actual_velocity_all(),r=right.get_actual_velocity_all();
    if(l.size()!=2||r.size()!=2){f.valid=false;return f;}
    for(int i=0;i<4;++i){double rpm=i<2?l[i]:r[i-2];f.wheel[i]=rpm*pi*diameter*ratio/60;if(!std::isfinite(rpm))f.valid=false;}
    return f;
}
lemlib::MotionResult followReference(pros::MotorGroup& left,pros::MotorGroup& right,
    double diameter,double ratio,const Config& config,double duration,double timeout,
    const std::function<Reference(double)>& reference,const std::function<bool()>& cancelled){
    using lemlib::MotionResult;
    if(!validConfig(config)||!std::isfinite(diameter)||diameter<=0||!std::isfinite(ratio)||ratio<=0||!std::isfinite(duration)||duration<0||!std::isfinite(timeout)||timeout<=0||timeout>120||!reference||!cancelled)return MotionResult::InvalidInput;
    if(cancelled())return MotionResult::Cancelled;
    auto& output=lemlib::driveOutput();auto token=output.acquire();
    if(!token)return lemlib::getOdomStatus().valid?MotionResult::Busy:MotionResult::SensorFault;
    struct Release{uint32_t token;~Release(){lemlib::driveOutput().release(token);}}release{token};
    Cascade control;control.config=config;
    const auto epoch=lemlib::getOdomStatus().epoch;
    const uint32_t start=pros::millis();uint32_t last=start,next=start,settleStart=0;bool settling=false;
    while(true){
        if(cancelled()||!output.owns(token))return MotionResult::Cancelled;
        auto status=lemlib::getOdomStatus();if(!status.valid||status.epoch!=epoch)return MotionResult::SensorFault;
        uint32_t now=pros::millis();double t=(now-start)*0.001,dt=(now-last)*0.001;last=now;
        if(t>=timeout)return MotionResult::TimedOut;
        if(dt==0)dt=0.01;
        auto feedback=readFeedback(left,right,diameter,ratio,epoch);auto target=reference(t);
        auto command=control.update(target,feedback,dt);
        if(!command.valid||!output.wheelVoltages(token,command.volts))return MotionResult::SensorFault;
        if(t>=duration&&settled(target,feedback)){
            if(!settling){settling=true;settleStart=now;}
            if(now-settleStart>=150)return MotionResult::Settled;
        }else settling=false;
        pros::Task::delay_until(&next,10);
    }
}
}
