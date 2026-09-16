#pragma once

#include "pros/adi.hpp"
#include "pros/distance.hpp"
#include "lemlib/chassis/chassis.hpp"
#include "lemlib/pose.hpp"

namespace lemlib {

/**
 * @brief Odometry recalibration and sensor fusion utilities.
 */
class OdomReset {
public:
    /**
     * @brief Reset X or Y coordinate using a physical bumper against a field perimeter wall
     */
    static void wallReset(Chassis& chassis, pros::adi::DigitalIn* bumper, bool isXAxis, float knownWallPositionInches);

    /**
     * @brief Reset X position when crossing the white center line or autonomous line
     */
    static void lineReset(Chassis& chassis, pros::adi::LineSensor* lineSensor, float lineXInches, int32_t reflectivityThreshold = 2000);

    /**
     * @brief Estimate global robot position using up to 4 VEX distance sensors.
     * Rotates sensor reading vectors based on current robot heading and ignores noisy/out-of-range readings.
     * 
     * @param chassis Reference to LemLib chassis
     * @param frontSensor Distance sensor facing front (can be nullptr)
     * @param backSensor Distance sensor facing back (can be nullptr)
     * @param leftSensor Distance sensor facing left (can be nullptr)
     * @param rightSensor Distance sensor facing right (can be nullptr)
     * @param fieldWidthInches Width of field (144.0 standard)
     * @param fieldHeightInches Height of field (144.0 standard)
     * @param maxValidDistanceInches Max valid range threshold (e.g. 50 inches)
     * @param autoApplyPose If true, immediately calls chassis.setPose() with filtered coordinates
     * @return Pose The estimated robot pose
     */
    static Pose distanceReset(Chassis& chassis,
                              pros::Distance* frontSensor,
                              pros::Distance* backSensor,
                              pros::Distance* leftSensor,
                              pros::Distance* rightSensor,
                              float fieldWidthInches = 144.0f,
                              float fieldHeightInches = 144.0f,
                              float maxValidDistanceInches = 60.0f,
                              bool autoApplyPose = true);
};

} // namespace lemlib
