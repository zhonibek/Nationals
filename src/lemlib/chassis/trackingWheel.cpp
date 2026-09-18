#include "lemlib/chassis/trackingWheel.hpp"
#include "lemlib/util.hpp"
#include <limits>
#include <cmath>
#include "pros/abstract_motor.hpp"
#include "pros/motor_group.hpp"
#include "pros/motors.h"

lemlib::TrackingWheel::TrackingWheel(pros::adi::Encoder* encoder, float wheelDiameter, float distance,
                                     float gearRatio) {
    this->encoder = encoder;
    this->diameter = wheelDiameter;
    this->distance = distance;
    this->gearRatio = gearRatio;
}

lemlib::TrackingWheel::TrackingWheel(pros::Rotation* encoder, float wheelDiameter, float distance, float gearRatio) {
    this->rotation = encoder;
    this->diameter = wheelDiameter;
    this->distance = distance;
    this->gearRatio = gearRatio;
}

lemlib::TrackingWheel::TrackingWheel(pros::MotorGroup* motors, float wheelDiameter, float distance, float rpm) {
    this->motors = motors;
    this->motors->set_encoder_units_all(pros::E_MOTOR_ENCODER_ROTATIONS);
    this->diameter = wheelDiameter;
    this->distance = distance;
    this->rpm = rpm;
}

void lemlib::TrackingWheel::reset() {
    if (this->encoder != nullptr) this->encoder->reset();
    if (this->rotation != nullptr) this->rotation->reset_position();
    if (this->motors != nullptr) this->motors->tare_position_all();
}

float lemlib::TrackingWheel::getDistanceTraveled() {
    if(!std::isfinite(diameter)||diameter<=0||!std::isfinite(gearRatio)||gearRatio<=0) return NAN;
    if (this->encoder != nullptr) {
        if(this->encoder->get_value()==PROS_ERR) return NAN;
        return (float(this->encoder->get_value()) * this->diameter * M_PI / 360) / this->gearRatio;
    } else if (this->rotation != nullptr) {
        if(this->rotation->get_position()==PROS_ERR) return NAN;
        return (float(this->rotation->get_position()) * this->diameter * M_PI / 36000) / this->gearRatio;
    } else if (this->motors != nullptr) {
        // get distance traveled by each motor
        std::vector<pros::MotorGears> gearsets = this->motors->get_gearing_all();
        std::vector<double> positions = this->motors->get_position_all();
        if(positions.empty()||positions.size()!=gearsets.size()) return NAN;
        std::vector<float> distances;
        for (int i = 0; i < this->motors->size(); i++) {
            float in;
            switch (gearsets[i]) {
                case pros::MotorGears::red: in = 100; break;
                case pros::MotorGears::green: in = 200; break;
                case pros::MotorGears::blue: in = 600; break;
                default: return NAN;
            }
            if(!std::isfinite(positions[i])) return NAN;
            distances.push_back(positions[i] * (diameter * M_PI) * (rpm / in));
        }
        return lemlib::avg(distances);
    } else {
        return 0;
    }
}

float lemlib::TrackingWheel::getOffset() { return this->distance; }

int lemlib::TrackingWheel::getType() {
    if (this->motors != nullptr) return 1;
    return 0;
}
