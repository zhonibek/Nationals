#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include "Point.hpp"

namespace pedro {

/**
 * @brief Bézier curve representation supporting linear, quadratic, cubic, and quintic curves.
 * Implements analytical derivatives, curvature, arc-length, and fast Newton-Raphson projection.
 */
class BezierCurve {
public:
    BezierCurve() = default;

    /**
     * @brief Construct a Cubic Bézier curve (standard Pedro Pathing primitive)
     */
    BezierCurve(const Point& p0, const Point& p1, const Point& p2, const Point& p3);

    /**
     * @brief Construct from arbitrary control points (linear, quadratic, cubic, quintic, etc.)
     */
    explicit BezierCurve(const std::vector<Point>& controlPoints);

    /**
     * @brief Evaluate point on the curve at parameter t in [0, 1]
     */
    Point getPoint(float t) const;

    /**
     * @brief Evaluate first derivative (tangent vector r'(t)) at parameter t
     */
    Vector2D getDerivative(float t) const;

    /**
     * @brief Evaluate second derivative (r''(t)) at parameter t
     */
    Vector2D getSecondDerivative(float t) const;

    /**
     * @brief Evaluate unit tangent vector T_hat(t)
     */
    Vector2D getTangent(float t) const;

    /**
     * @brief Evaluate unit normal vector N_hat(t) pointing perpendicular to tangent (to the left)
     */
    Vector2D getNormal(float t) const;

    /**
     * @brief Evaluate curvature kappa(t) = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)
     */
    float getCurvature(float t) const;

    /**
     * @brief Total arc length of curve
     */
    float getLength(int samples = 40) const;

    /**
     * @brief Arc length from parameter t to the end of the curve (t = 1.0)
     */
    float getRemainingDistance(float t, int samples = 25) const;

    /**
     * @brief Fast, robust projection of robot pose onto the curve.
     * Finds parameter t in [0, 1] minimizing Euclidean distance to the curve.
     * Combines coarse sample grid search with Newton-Raphson root refinement.
     */
    float project(const Point& robotPose, float tGuess = 0.0f) const;

    /**
     * @brief Get reference to control points
     */
    const std::vector<Point>& getControlPoints() const { return controlPoints; }

    Point getStartPoint() const { return controlPoints.empty() ? Point() : controlPoints.front(); }
    Point getEndPoint() const { return controlPoints.empty() ? Point() : controlPoints.back(); }

private:
    std::vector<Point> controlPoints;
};

} // namespace pedro
