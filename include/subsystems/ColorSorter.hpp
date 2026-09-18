#pragma once
#include <atomic>
#include "lemlib/safety.hpp"

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
    // This subsystem is the sole writer of its intake motor.
    void setIntakePower(int power);

private:
    pros::Optical* optical;
    pros::MotorGroup* motor;
    std::atomic<AllianceColor> alliance;
    pros::Mutex outputMutex;
    int intakePower=0;
    std::atomic<bool> enabled{true};
    std::atomic<bool> running{false};
    pros::Mutex lifecycleMutex;
    uint32_t checkPeriodMs = 10;
    pros::Task* task = nullptr;

    uint32_t ejectStartTime = 0;
    bool isEjecting = false;

    static void task_fn(void* param);
};

} // namespace lemlib
