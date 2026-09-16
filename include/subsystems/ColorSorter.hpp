#pragma once

#include "pros/rtos.hpp"
#include "pros/optical.hpp"
#include "pros/motor_group.hpp"

namespace lemlib {

enum class AllianceColor {
    RED,
    BLUE,
    DISABLED
};

enum class DetectedColor {
    NONE,
    RED,
    BLUE
};

/**
 * @brief Autonomous and driver-assist Color Sorter.
 * Uses a VEX Optical sensor to detect game pieces in the intake/indexer.
 * Automatically reverses or stops the top roller / intake to eject opposing alliance blocks.
 */
class ColorSorter {
public:
    ColorSorter(pros::Optical* opticalSensor, pros::MotorGroup* sortMotor, AllianceColor targetAlliance = AllianceColor::RED);
    ~ColorSorter();

    void startTask(uint32_t periodMs = 10);
    void stopTask();

    void setAlliance(AllianceColor alliance);
    AllianceColor getAlliance() const;

    void setEnabled(bool enabled);
    bool isEnabled() const;

    DetectedColor detectColor();

    /**
     * @brief Sort step executed in loop or task
     */
    void update();

private:
    pros::Optical* optical;
    pros::MotorGroup* motor;
    AllianceColor alliance;
    bool enabled = true;
    bool running = false;
    uint32_t checkPeriodMs = 10;
    pros::Task* task = nullptr;

    uint32_t ejectStartTime = 0;
    bool isEjecting = false;

    static void task_fn(void* param);
};

} // namespace lemlib
