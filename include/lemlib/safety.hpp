#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include "pros/rtos.hpp"
#include "pros/motor_group.hpp"

namespace lemlib {
class Lock {
    pros::Mutex& mutex;
public:
    explicit Lock(pros::Mutex& m) : mutex(m) { mutex.take(TIMEOUT_MAX); }
    ~Lock() { mutex.give(); }
    Lock(const Lock&) = delete;
    Lock& operator=(const Lock&) = delete;
};
enum class MotionResult { Idle, Running, Settled, Chained, Cancelled, TimedOut, SensorFault, InvalidInput, Busy };
// Exactly one lease may write the drivetrain. A revoked token can never restart it.
// All commands use power units [-127,127]; conversion happens only after validation.
class DriveOutput {
public:
    bool configure(pros::MotorGroup* left, pros::MotorGroup* right);
    uint32_t acquire(bool requireOdometry = true);
    bool tank(uint32_t token, float left, float right);
    bool holonomic(uint32_t token, float forward, float right, float clockwise, float cap = 127);
    bool wheels(uint32_t token, float fl, float bl, float fr, float br, float cap = 127);
    void release(uint32_t token);
    void stop();
    void watchdog();
    bool owns(uint32_t token);
private:
    bool allowed(uint32_t token);
    void zero();
    pros::Mutex mutex;
    pros::MotorGroup* left = nullptr;
    pros::MotorGroup* right = nullptr;
    uint32_t sequence = 0, owner = 0, lastWrite = 0;
    int mode = 0;
    bool feedback = true;
};
DriveOutput& driveOutput();
// Measured seconds, with the first sample at the nominal period.
class LoopClock {
    uint32_t last = pros::millis();
public:
    float tick() { auto now=pros::millis(); float dt=(now-last)*0.001f; last=now; return dt>0 ? dt : 0.01f; }
};
}
