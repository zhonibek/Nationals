#include "subsystems/ColorSorter.hpp"
#include <cmath>
#include "pros/misc.hpp"

namespace lemlib {

ColorSorter::ColorSorter(pros::Optical* opticalSensor, pros::MotorGroup* sortMotor, AllianceColor targetAlliance)
    : optical(opticalSensor),
      motor(sortMotor),
      alliance(targetAlliance) {
    if (optical) {
        optical->set_led_pwm(100); // turn on optical LED for consistent color sensing
    }
}

ColorSorter::~ColorSorter() {
    stopTask();
}

void ColorSorter::startTask(uint32_t periodMs) {
    Lock lifecycle(lifecycleMutex);
    if (running) return;
    checkPeriodMs = std::clamp<uint32_t>(periodMs,5u,1000u);
    running = true;
    task = new pros::Task(task_fn, this, "ColorSortTask");
}

void ColorSorter::stopTask() {
    Lock lifecycle(lifecycleMutex);
    running = false;
    if (task) {
        task->join();
        delete task;
        task = nullptr;
    }
    Lock guard(outputMutex); isEjecting=false; if(motor) motor->move(0);
}

void ColorSorter::setAlliance(AllianceColor newAlliance) {
    alliance = newAlliance;
    if(newAlliance==AllianceColor::DISABLED) {Lock guard(outputMutex);isEjecting=false;if(motor) motor->move(0);}
}

AllianceColor ColorSorter::getAlliance() const {
    return alliance;
}

void ColorSorter::setEnabled(bool isEnabled) {
    enabled = isEnabled;
    if(!isEnabled) {Lock guard(outputMutex);isEjecting=false;if(motor) motor->move(0);}
}

bool ColorSorter::isEnabled() const {
    return enabled;
}

DetectedColor ColorSorter::detectColor() {
    if (!optical) return DetectedColor::NONE;

    // Check proximity to ensure a piece is actually in the intake
    int32_t proximity = optical->get_proximity();
    if (proximity == PROS_ERR || proximity < 120) {
        return DetectedColor::NONE;
    }

    double hue = optical->get_hue();
    // Red hue spans [0, 30] and [330, 360]
    if ((hue >= 0.0 && hue <= 35.0) || (hue >= 325.0 && hue <= 360.0)) {
        return DetectedColor::RED;
    }
    // Blue hue spans [190, 260]
    if (hue >= 180.0 && hue <= 260.0) {
        return DetectedColor::BLUE;
    }

    return DetectedColor::NONE;
}

void ColorSorter::task_fn(void* param) {
    auto* self = static_cast<ColorSorter*>(param);
    while (self->running) {
        self->update();
        pros::delay(self->checkPeriodMs);
    }
}

void ColorSorter::setIntakePower(int power) { Lock guard(outputMutex); intakePower=std::clamp(power,-127,127); }

void ColorSorter::update() {
    Lock guard(outputMutex);
    if (pros::competition::is_disabled() || !enabled || alliance == AllianceColor::DISABLED || !optical || !motor) {
        isEjecting=false; if(motor) motor->move(0);
        return;
    }

    uint32_t now = pros::millis();

    // Handle ongoing ejection window
    if (isEjecting) {
        if (now - ejectStartTime < 250) {
            motor->move(-127); // eject piece out back or top
            return;
        } else {
            isEjecting = false;
        }
    }

    motor->move(intakePower);
    DetectedColor detected = detectColor();

    if (detected != DetectedColor::NONE) {
        bool isOpponent = false;
        if (alliance == AllianceColor::RED && detected == DetectedColor::BLUE) {
            isOpponent = true;
        } else if (alliance == AllianceColor::BLUE && detected == DetectedColor::RED) {
            isOpponent = true;
        }

        if (isOpponent) {
            isEjecting = true;
            ejectStartTime = now;
            motor->move(-127); // Eject
        }
    }
}

} // namespace lemlib
