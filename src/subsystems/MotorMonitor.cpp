#include <cmath>
#include "subsystems/MotorMonitor.hpp"
#include "pros/error.h"
#include <iostream>

namespace lemlib {

MotorMonitor::MotorMonitor(pros::Controller& controller,
                           const std::vector<std::pair<std::string, pros::MotorGroup*>>& motorGroups,
                           const std::vector<std::pair<std::string, pros::Distance*>>& distanceSensors,
                           float tempWarningCelsius)
    : controller(controller),
      motorGroups(motorGroups),
      distanceSensors(distanceSensors),
      tempThreshold(tempWarningCelsius) {}

MotorMonitor::~MotorMonitor() {
    stopTask();
}

void MotorMonitor::startTask(uint32_t periodMs) {
    if (running) return;
    checkPeriodMs = periodMs;
    running = true;
    task = new pros::Task(task_fn, this, "MotorWatchdog");
}

void MotorMonitor::stopTask() {
    if (!running) return;
    running = false;
    if (task) {
        task->remove();
        delete task;
        task = nullptr;
    }
}

void MotorMonitor::task_fn(void* param) {
    auto* self = static_cast<MotorMonitor*>(param);
    while (self->running) {
        self->check();
        pros::delay(self->checkPeriodMs);
    }
}

void MotorMonitor::check() {
    uint32_t now = pros::millis();
    std::string alertMessage = "";
    bool isDisconnect = false;
    bool isOvertemp = false;

    // Check Motor Groups
    for (const auto& [name, group] : motorGroups) {
        if (!group) continue;
        auto temps = group->get_temperature_all();
        auto voltages = group->get_voltage_all();

        for (size_t i = 0; i < temps.size(); ++i) {
            // Disconnect check (PROS returns PROS_ERR / INT32_MAX on disconnected motors)
            if (voltages[i] == INT32_MIN || voltages[i] == PROS_ERR) {
                alertMessage = "DISCONN: " + name + "[" + std::to_string(i) + "]";
                isDisconnect = true;
                break;
            }

            // Overtemp check
            if (temps[i] >= tempThreshold && temps[i] != PROS_ERR_F && !std::isinf(temps[i])) {
                alertMessage = "OVERTEMP " + std::to_string((int)temps[i]) + "C: " + name;
                isOvertemp = true;
                break;
            }
        }
        if (isDisconnect || isOvertemp) break;
    }

    // Check Distance Sensors
    if (!isDisconnect && !isOvertemp) {
        for (const auto& [name, distSensor] : distanceSensors) {
            if (!distSensor) continue;
            int32_t dist = distSensor->get();
            if (dist == PROS_ERR) {
                alertMessage = "DISCONN DIST: " + name;
                isDisconnect = true;
                break;
            }
        }
    }

    // Deliver alert to controller LCD (throttled to at most once per 2 seconds)
    if (!alertMessage.empty() && (now - lastAlertTime > 2000)) {
        lastAlertTime = now;
        controller.print(1, 0, "%-18s", alertMessage.c_str());

        if (isDisconnect) {
            controller.rumble("---"); // long rumble for critical disconnect
        } else if (isOvertemp) {
            controller.rumble("..");  // double buzz for thermal warning
        }
    }
}

} // namespace lemlib
