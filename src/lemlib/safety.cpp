#include "lemlib/safety.hpp"
#include "lemlib/chassis/odom.hpp"
#include "pros/misc.hpp"
#include "pros/motors.h"

namespace lemlib {
DriveOutput& driveOutput() { static DriveOutput output; return output; }
bool DriveOutput::configure(pros::MotorGroup* l, pros::MotorGroup* r) {
    Lock guard(mutex);
    if (!l || !r || l->size()==0 || r->size()==0) return false;
    if (owner && (l!=left || r!=right)) return false;
    left=l; right=r; return true;
}
void DriveOutput::zero() {
    if(left) { left->move(0); left->brake(); }
    if(right) { right->move(0); right->brake(); }
}
uint32_t DriveOutput::acquire(bool requireOdometry) {
    Lock guard(mutex);
    if(owner || !left || !right || pros::competition::is_disabled()) return 0;
    if(requireOdometry && !getOdomStatus().valid) return 0;
    if(++sequence==0) ++sequence;
    owner=sequence; feedback=requireOdometry; mode=pros::competition::get_status();
    lastWrite=pros::millis(); return owner;
}
bool DriveOutput::allowed(uint32_t token) {
    if(!token || token!=owner) return false;
    if(pros::competition::is_disabled() || mode!=pros::competition::get_status() ||
       (feedback && !getOdomStatus().valid)) { zero(); owner=0; return false; }
    return true;
}
bool DriveOutput::owns(uint32_t token) { Lock guard(mutex); return allowed(token); }
bool DriveOutput::tank(uint32_t token,float l,float r) {
    Lock guard(mutex);
    if(!allowed(token)) return false;
    if(!std::isfinite(l) || !std::isfinite(r)) { zero(); owner=0; return false; }
    const int a=static_cast<int>(std::clamp(l,-127.f,127.f)), b=static_cast<int>(std::clamp(r,-127.f,127.f));
    bool ok=left->move(a)!=PROS_ERR && right->move(b)!=PROS_ERR;
    if(!ok) { zero(); owner=0; }
    lastWrite=pros::millis(); return ok;
}
bool DriveOutput::wheels(uint32_t token,float fl,float bl,float fr,float br,float cap) {
    Lock guard(mutex);
    if(!allowed(token)) return false;
    if(!std::isfinite(fl)||!std::isfinite(bl)||!std::isfinite(fr)||!std::isfinite(br)||
       !std::isfinite(cap)||cap<=0||left->size()!=2||right->size()!=2) { zero(); owner=0; return false; }
    cap=std::min(cap,127.f);
    const float scale=std::max({1.f,std::fabs(fl)/cap,std::fabs(bl)/cap,std::fabs(fr)/cap,std::fabs(br)/cap});
    auto lp=left->get_port_all(), rp=right->get_port_all();
    bool ok=true;
    const float commands[]={fl,bl,fr,br};
    const int ports[]={lp[0],lp[1],rp[0],rp[1]};
    for(int i=0;i<4;++i) ok=(pros::c::motor_move(ports[i],static_cast<int>(commands[i]/scale))!=PROS_ERR)&&ok;
    if(!ok) { zero(); owner=0; }
    lastWrite=pros::millis(); return ok;
}
bool DriveOutput::holonomic(uint32_t token,float f,float s,float t,float cap) {
    return wheels(token,f+s+t,f-s+t,f-s-t,f+s-t,cap);
}
void DriveOutput::release(uint32_t token) { Lock guard(mutex); if(token && owner==token) { zero(); owner=0; } }
void DriveOutput::stop() { Lock guard(mutex); zero(); owner=0; }
void DriveOutput::watchdog() {
    Lock guard(mutex);
    if(owner && (!allowed(owner) || static_cast<uint32_t>(pros::millis()-lastWrite)>100)) { zero(); owner=0; }
}
}
