#pragma once

#include <vector>
#include <string>
#include "pros/rtos.hpp"
#include "pros/misc.hpp"
#include "pros/motor_group.hpp"
#include "pros/distance.hpp"

namespace lemlib {

/**
 * @brief Background watchdog for robot motors and sensors.
 * Monitors motor disconnects, over-temperature, and sensor health.
 * Alerts driver via Controller LCD display and haptic rumble patterns.
 */
class MotorMonitor {
public:
    MotorMonitor(pros::Controller& controller,
                 const std::vector<std::pair<std::string, pros::MotorGroup*>>& motorGroups,
                 const std::vector<std::pair<std::string, pros::Distance*>>& distanceSensors = {},
                 float tempWarningCelsius = 55.0f);

    ~MotorMonitor();

    void startTask(uint32_t periodMs = 500);
    void stopTask();

    void check();

private:
    pros::Controller& controller;
    std::vector<std::pair<std::string, pros::MotorGroup*>> motorGroups;
    std::vector<std::pair<std::string, pros::Distance*>> distanceSensors;
    float tempThreshold;

    pros::Task* task = nullptr;
    bool running = false;
    uint32_t checkPeriodMs = 500;
    uint32_t lastAlertTime = 0;

    static void task_fn(void* param);
};

} // namespace lemlib
