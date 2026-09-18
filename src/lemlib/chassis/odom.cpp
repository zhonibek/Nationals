#include "lemlib/chassis/odom.hpp"
#include "lemlib/safety.hpp"
#include "lemlib/util.hpp"
#include <array>
#include <cmath>
#include <limits>

namespace {
pros::Mutex mutex;
pros::Task* task=nullptr;
lemlib::OdomSensors sensors(nullptr,nullptr,nullptr,nullptr,nullptr);
lemlib::Drivetrain drive(nullptr,nullptr,0,0,0,0);
lemlib::DrivebaseType type=lemlib::DrivebaseType::TANK;
lemlib::Pose pose(0,0,0), speed(0,0,0), local(0,0,0);
std::array<float,5> previous{};
uint32_t sampleTime=0, epoch=0;
bool initialized=false, fault=false;
lemlib::OdomFault reason=lemlib::OdomFault::Uninitialized;
bool readSample(std::array<float,5>& sample) {
    sample.fill(0);
    if(type==lemlib::DrivebaseType::MECANUM) { reason=lemlib::OdomFault::UnsupportedDrive; return false; }
    if(!drive.leftMotors||!drive.rightMotors||!std::isfinite(drive.trackWidth)||drive.trackWidth<=0 ||
       !std::isfinite(drive.wheelDiameter)||drive.wheelDiameter<=0||!std::isfinite(drive.rpm)||drive.rpm<=0) {
        reason=lemlib::OdomFault::Configuration; return false;
    }
    if(type==lemlib::DrivebaseType::XDRIVE) {
        auto l=drive.leftMotors->get_position_all(), r=drive.rightMotors->get_position_all();
        auto lg=drive.leftMotors->get_gearing_all(), rg=drive.rightMotors->get_gearing_all();
        if(l.size()!=2||r.size()!=2||lg.size()!=2||rg.size()!=2) { reason=lemlib::OdomFault::Configuration; return false; }
        for(int i=0;i<4;++i) {
            auto gear=i<2 ? lg[i] : rg[i-2];
            float rpm=gear==pros::MotorGears::red ? 100 : gear==pros::MotorGears::green ? 200 : gear==pros::MotorGears::blue ? 600 : 0;
            if(rpm==0) { reason=lemlib::OdomFault::Sensor; return false; }
            sample[i]=(i<2?l[i]:r[i-2])*drive.wheelDiameter*M_PI*drive.rpm/rpm;
        }
    } else {
        lemlib::TrackingWheel* wheels[]={sensors.vertical1,sensors.vertical2,sensors.horizontal1,sensors.horizontal2};
        for(int i=0;i<4;++i) if(wheels[i]) sample[i]=wheels[i]->getDistanceTraveled();
        if(!sensors.vertical1 && !sensors.vertical2) { reason=lemlib::OdomFault::Configuration; return false; }
    }
    if(sensors.imu) {
        if(sensors.imu->is_calibrating()) { reason=lemlib::OdomFault::Sensor; return false; }
        sample[4]=lemlib::degToRad(sensors.imu->get_rotation());
    }
    for(float value:sample) if(!std::isfinite(value)) { reason=lemlib::OdomFault::Sensor; return false; }
    return true;
}
void rebase() {
    speed={0,0,0}; local={0,0,0}; ++epoch;
    initialized=readSample(previous); fault=!initialized;
    sampleTime=pros::millis();
    if(initialized) reason=lemlib::OdomFault::None;
}
void fail(lemlib::OdomFault why) { fault=true; reason=why; initialized=false; speed={0,0,0}; local={0,0,0}; }
}
void lemlib::setSensors(OdomSensors s,Drivetrain d) {
    Lock guard(mutex); sensors=s; drive=d;
    if(drive.leftMotors) drive.leftMotors->set_encoder_units_all(pros::E_MOTOR_ENCODER_ROTATIONS);
    if(drive.rightMotors) drive.rightMotors->set_encoder_units_all(pros::E_MOTOR_ENCODER_ROTATIONS);
    rebase();
}
void lemlib::setDrivebaseType(DrivebaseType t) { Lock guard(mutex); type=t; rebase(); }
lemlib::DrivebaseType lemlib::getDrivebaseType() { Lock guard(mutex); return type; }
lemlib::OdomStatus lemlib::getOdomStatus() {
    Lock guard(mutex);
    const auto age=pros::millis()-sampleTime;
    return {initialized && !fault && age<=100, reason, sampleTime, epoch};
}
lemlib::OdomSnapshot lemlib::getOdomSnapshot() {
    Lock guard(mutex);
    return {pose,speed,local,{initialized&&!fault&&pros::millis()-sampleTime<=100,reason,sampleTime,epoch}};
}
lemlib::Pose lemlib::getPose(bool radians) {
    Lock guard(mutex); auto p=pose; if(!radians) p.theta=radToDeg(p.theta); return p;
}
lemlib::Pose lemlib::getSpeed(bool radians) {
    Lock guard(mutex); auto p=speed; if(!radians) p.theta=radToDeg(p.theta); return p;
}
lemlib::Pose lemlib::getLocalSpeed(bool radians) {
    Lock guard(mutex); auto p=local; if(!radians) p.theta=radToDeg(p.theta); return p;
}
void lemlib::setPose(Pose p,bool radians) {
    Lock guard(mutex);
    if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.theta)) { fail(OdomFault::InvalidPose); return; }
    pose=p; if(!radians) pose.theta=degToRad(p.theta); rebase();
}
lemlib::Pose lemlib::estimatePose(float time,bool radians) {
    const auto s=getOdomSnapshot(); auto p=s.pose;
    if(std::isfinite(time)&&time>=0&&s.status.valid) {
        const float dh=s.localSpeed.theta*time, mid=p.theta+dh/2;
        const float scale=std::fabs(dh)<1e-5f ? 1.f : 2.f*std::sin(dh/2)/dh;
        const float right=s.localSpeed.x*time*scale, forward=s.localSpeed.y*time*scale;
        p.x+=forward*std::sin(mid)+right*std::cos(mid);
        p.y+=forward*std::cos(mid)-right*std::sin(mid); p.theta+=dh;
    }
    if(!radians) p.theta=radToDeg(p.theta);
    return p;
}
void lemlib::update() {
    Lock guard(mutex);
    // Faults latch: explicit setPose/calibrate is required to accept a new encoder epoch.
    if(fault) return;
    std::array<float,5> sample{};
    if(!readSample(sample)) { fail(reason); return; }
    const auto now=pros::millis();
    if(!initialized) { previous=sample; sampleTime=now; initialized=true; reason=OdomFault::None; return; }
    const float dt=(now-sampleTime)*0.001f;
    if(dt<=0) return;
    if(dt>0.1f) { fail(OdomFault::Stale); return; }
    std::array<float,5> d{};
    for(int i=0;i<5;++i) {
        d[i]=sample[i]-previous[i];
        if(std::fabs(d[i]) > (i==4?50.f:300.f)*dt) { fail(OdomFault::Discontinuity); return; }
    }
    float dx=0,dy=0,dh=0;
    if(type==DrivebaseType::XDRIVE) {
        dx=(d[0]-d[1]-d[2]+d[3])/4; dy=(d[0]+d[1]+d[2]+d[3])/4;
        dh=sensors.imu ? d[4] : (d[0]+d[1]-d[2]-d[3])/(2*drive.trackWidth);
    } else {
        bool heading=false;
        auto pairHeading=[&](TrackingWheel* a,TrackingWheel* b,float da,float db) {
            if(a&&b&&std::isfinite(a->getOffset())&&std::isfinite(b->getOffset())&&std::fabs(a->getOffset()-b->getOffset())>1e-4f) {
                dh=-(da-db)/(a->getOffset()-b->getOffset()); heading=true;
            }
        };
        pairHeading(sensors.horizontal1,sensors.horizontal2,d[2],d[3]);
        if(!heading&&sensors.vertical1&&sensors.vertical2&&!sensors.vertical1->getType()&&!sensors.vertical2->getType())
            pairHeading(sensors.vertical1,sensors.vertical2,d[0],d[1]);
        if(!heading&&sensors.imu) { dh=d[4]; heading=true; }
        if(!heading) pairHeading(sensors.vertical1,sensors.vertical2,d[0],d[1]);
        if(!heading) { fail(OdomFault::Configuration); return; }
        int v=sensors.vertical1&&(!sensors.vertical1->getType()||!sensors.vertical2) ? 0 : 1;
        auto vw=v==0?sensors.vertical1:sensors.vertical2;
        int h=sensors.horizontal1?2:3; auto hw=h==2?sensors.horizontal1:sensors.horizontal2;
        const float chord=std::fabs(dh)<1e-5f ? 1.f : 2*std::sin(dh/2)/dh;
        dy=vw ? (d[v]+dh*vw->getOffset())*chord : 0;
        // Legacy horizontal wheel distance is positive left. Public local.x is always right.
        dx=hw ? -(d[h]+dh*hw->getOffset())*chord : 0;
    }
    if(!std::isfinite(dx)||!std::isfinite(dy)||!std::isfinite(dh)) { fail(OdomFault::Sensor); return; }
    const float mid=pose.theta+dh/2;
    const float gx=dy*std::sin(mid)+dx*std::cos(mid), gy=dy*std::cos(mid)-dx*std::sin(mid);
    pose.x+=gx; pose.y+=gy; pose.theta+=dh;
    const float alpha=1-std::exp(-dt/0.025f);
    speed={ema(gx/dt,speed.x,alpha),ema(gy/dt,speed.y,alpha),ema(dh/dt,speed.theta,alpha)};
    local={ema(dx/dt,local.x,alpha),ema(dy/dt,local.y,alpha),speed.theta};
    previous=sample; sampleTime=now; reason=OdomFault::None;
}
void lemlib::init() {
    if(!task) task=new pros::Task([] {
        uint32_t next=pros::millis();
        while(true) { update(); driveOutput().watchdog(); pros::Task::delay_until(&next,10); }
    });
}
