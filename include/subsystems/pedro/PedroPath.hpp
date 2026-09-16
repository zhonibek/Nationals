#pragma once

#include <vector>
#include <memory>
#include "BezierCurve.hpp"

namespace pedro {

enum class HeadingMode {
    TANGENT,      // Align robot heading with curve tangent vector
    CONSTANT,     // Hold fixed heading throughout trajectory
    FACING_POINT, // Continuously aim at a designated point on the field (e.g. goal / goal stake)
    LINEAR        // Linearly interpolate between start heading and end heading
};

/**
 * @brief Path container holding Bézier curve segments with decoupled heading specifications
 */
class PedroPath {
public:
    PedroPath() = default;

    /**
     * @brief Construct a single-curve path
     */
    explicit PedroPath(const BezierCurve& curve, HeadingMode mode = HeadingMode::TANGENT, float targetHeading = 0.0f)
        : headingMode(mode), constantHeading(targetHeading) {
        curves.push_back(curve);
    }

    /**
     * @brief Add a Bézier curve segment to the path
     */
    void addSegment(const BezierCurve& curve) {
        curves.push_back(curve);
    }

    void setHeadingMode(HeadingMode mode) { headingMode = mode; }
    HeadingMode getHeadingMode() const { return headingMode; }

    void setConstantHeading(float deg) {
        headingMode = HeadingMode::CONSTANT;
        constantHeading = deg;
    }

    void setFacingPoint(const Point& point) {
        headingMode = HeadingMode::FACING_POINT;
        targetFacingPoint = point;
    }

    void setLinearHeading(float startDeg, float endDeg) {
        headingMode = HeadingMode::LINEAR;
        startHeading = startDeg;
        endHeading = endDeg;
    }

    const std::vector<BezierCurve>& getSegments() const { return curves; }
    size_t getSegmentCount() const { return curves.size(); }

    /**
     * @brief Get target heading (degrees) for a given curve segment and parameter t
     */
    float getTargetHeading(size_t segmentIdx, float t, const Point& robotPos) const {
        if (segmentIdx >= curves.size()) return constantHeading;

        switch (headingMode) {
            case HeadingMode::CONSTANT:
                return constantHeading;

            case HeadingMode::FACING_POINT: {
                // Vector pointing from robot to target point
                float dx = targetFacingPoint.x - robotPos.x;
                float dy = targetFacingPoint.y - robotPos.y;
                // LemLib coordinate frame: 0 deg is North (+Y), 90 deg is East (+X)
                float angleRad = std::atan2(dx, dy);
                float angleDeg = angleRad * (180.0f / M_PI);
                return angleDeg;
            }

            case HeadingMode::LINEAR: {
                return startHeading + (endHeading - startHeading) * t;
            }

            case HeadingMode::TANGENT:
            default: {
                Vector2D tangent = curves[segmentIdx].getTangent(t);
                // LemLib: theta = atan2(dx, dy)
                float angleRad = std::atan2(tangent.x, tangent.y);
                float angleDeg = angleRad * (180.0f / M_PI);
                return angleDeg;
            }
        }
    }

private:
    std::vector<BezierCurve> curves;
    HeadingMode headingMode = HeadingMode::TANGENT;
    float constantHeading = 0.0f;
    float startHeading = 0.0f;
    float endHeading = 0.0f;
    Point targetFacingPoint{0.0f, 0.0f};
};

} // namespace pedro
