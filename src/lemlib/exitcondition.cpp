#include <cmath>
#include "pros/rtos.hpp"
#include "lemlib/exitcondition.hpp"

namespace lemlib {
ExitCondition::ExitCondition(const float range, const int time)
    : range(range),
      time(time) {}

bool ExitCondition::getExit() { return done; }

bool ExitCondition::update(const float input) {
    const uint32_t curTime = pros::millis();
    if (!std::isfinite(input) || !std::isfinite(range) || range<0 || time<0 || std::fabs(input)>range) {
        timing=false; done=false;
    } else {
        if(!timing) { startTime=curTime; timing=true; }
        done=curTime-startTime>=static_cast<uint32_t>(time);
    }
    return done;
}

void ExitCondition::reset() {
    startTime = 0;
    timing=false;
    done = false;
}
} // namespace lemlib