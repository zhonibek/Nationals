#include "subsystems/OdomReset.hpp"
#include <cmath>
#include <vector>

namespace lemlib {

void OdomReset::wallReset(Chassis& chassis, pros::adi::DigitalIn* bumper, bool isXAxis, float knownWallPositionInches) {
    if (!bumper) return;
    if (bumper->get_value()) { // Pressed against wall
        Pose current = chassis.getPose();
        if (isXAxis) {
            chassis.setPose(knownWallPositionInches, current.y, current.theta);
        } else {
            chassis.setPose(current.x, knownWallPositionInches, current.theta);
        }
    }
}

void OdomReset::lineReset(Chassis& chassis, pros::adi::LineSensor* lineSensor, float lineXInches, int32_t reflectivityThreshold) {
    if (!lineSensor) return;
    // Reflectivity dips when crossing white tape on dark foam tiles
    int32_t val = lineSensor->get_value();
    if (val < reflectivityThreshold) {
        Pose current = chassis.getPose();
        chassis.setPose(lineXInches, current.y, current.theta);
    }
}

Pose OdomReset::distanceReset(Chassis& chassis,
                              pros::Distance* frontSensor,
                              pros::Distance* backSensor,
                              pros::Distance* leftSensor,
                              pros::Distance* rightSensor,
                              float fieldWidthInches,
                              float fieldHeightInches,
                              float maxValidDistanceInches,
                              bool autoApplyPose) {
    Pose current = chassis.getPose(true); // heading in radians
    double theta = current.theta;

    double cosT = std::cos(theta);
    double sinT = std::sin(theta);

    std::vector<double> x_estimates;
    std::vector<double> y_estimates;

    // Helper lambda to check and convert sensor reading
    auto processSensor = [&](pros::Distance* sensor, double localAngleOffsetRad) {
        if (!sensor) return;
        int32_t raw_mm = sensor->get();
        if (raw_mm <= 0 || raw_mm == PROS_ERR) return;
        double distInches = raw_mm / 25.4;
        if (distInches > maxValidDistanceInches || distInches < 2.0) return;

        double sensorHeading = theta + localAngleOffsetRad;
        double dx = std::sin(sensorHeading);
        double dy = std::cos(sensorHeading);

        // If sensor is aligned near field cardinal directions (tight tolerance < 8 deg)
        if (std::abs(dx) > 0.98) { // Facing east or west
            double projectedDist = distInches * std::abs(dx);
            if (dx > 0) { // Facing East (+X wall at +fieldWidth/2)
                x_estimates.push_back((fieldWidthInches / 2.0) - projectedDist);
            } else { // Facing West (-X wall at -fieldWidth/2)
                x_estimates.push_back((-fieldWidthInches / 2.0) + projectedDist);
            }
        }
        if (std::abs(dy) > 0.98) { // Facing north or south
            double projectedDist = distInches * std::abs(dy);
            if (dy > 0) { // Facing North (+Y wall at +fieldHeight/2)
                y_estimates.push_back((fieldHeightInches / 2.0) - projectedDist);
            } else { // Facing South (-Y wall at -fieldHeight/2)
                y_estimates.push_back((-fieldHeightInches / 2.0) + projectedDist);
            }
        }
    };

    processSensor(frontSensor, 0.0);
    processSensor(rightSensor, M_PI_2);
    processSensor(backSensor, M_PI);
    processSensor(leftSensor, -M_PI_2);

    double newX = current.x;
    double newY = current.y;

    if (!x_estimates.empty()) {
        double sumX = 0.0;
        for (double v : x_estimates) sumX += v;
        newX = sumX / x_estimates.size();
    }

    if (!y_estimates.empty()) {
        double sumY = 0.0;
        for (double v : y_estimates) sumY += v;
        newY = sumY / y_estimates.size();
    }

    Pose result(newX, newY, chassis.getPose().theta);

    if (autoApplyPose && (!x_estimates.empty() || !y_estimates.empty())) {
        chassis.setPose(result);
    }

    return result;
}

} // namespace lemlib
