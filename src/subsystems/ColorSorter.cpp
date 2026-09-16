#include "subsystems/ColorSorter.hpp"
#include <cmath>

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
    if (running) return;
    checkPeriodMs = periodMs;
    running = true;
    task = new pros::Task(task_fn, this, "ColorSortTask");
}

void ColorSorter::stopTask() {
    if (!running) return;
    running = false;
    if (task) {
        task->remove();
        delete task;
        task = nullptr;
    }
}

void ColorSorter::setAlliance(AllianceColor newAlliance) {
    alliance = newAlliance;
}

AllianceColor ColorSorter::getAlliance() const {
    return alliance;
}

void ColorSorter::setEnabled(bool isEnabled) {
    enabled = isEnabled;
}

bool ColorSorter::isEnabled() const {
    return enabled;
}

DetectedColor ColorSorter::detectColor() {
    if (!optical) return DetectedColor::NONE;

    // Check proximity to ensure a piece is actually in the intake
    int32_t proximity = optical->get_proximity();
    if (proximity < 120) {
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

void ColorSorter::update() {
    if (!enabled || alliance == AllianceColor::DISABLED || !optical || !motor) {
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
