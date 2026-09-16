#include "subsystems/pedro/BezierCurve.hpp"

namespace pedro {

BezierCurve::BezierCurve(const Point& p0, const Point& p1, const Point& p2, const Point& p3) {
    controlPoints = {p0, p1, p2, p3};
}

BezierCurve::BezierCurve(const std::vector<Point>& points) : controlPoints(points) {}

Point BezierCurve::getPoint(float t) const {
    t = std::clamp(t, 0.0f, 1.0f);
    size_t n = controlPoints.size();
    if (n == 0) return Point(0.0f, 0.0f);
    if (n == 1) return controlPoints[0];

    // Optimized evaluation for standard Cubic Bézier (4 control points)
    if (n == 4) {
        float u = 1.0f - t;
        float u2 = u * u;
        float u3 = u2 * u;
        float t2 = t * t;
        float t3 = t2 * t;

        return controlPoints[0] * u3 +
               controlPoints[1] * (3.0f * u2 * t) +
               controlPoints[2] * (3.0f * u * t2) +
               controlPoints[3] * t3;
    }

    // Optimized evaluation for Linear (2 control points)
    if (n == 2) {
        return controlPoints[0] * (1.0f - t) + controlPoints[1] * t;
    }

    // General De Casteljau algorithm for arbitrary degrees
    std::vector<Point> temp = controlPoints;
    for (size_t step = 1; step < n; ++step) {
        for (size_t i = 0; i < n - step; ++i) {
            temp[i] = temp[i] * (1.0f - t) + temp[i + 1] * t;
        }
    }
    return temp[0];
}

Vector2D BezierCurve::getDerivative(float t) const {
    t = std::clamp(t, 0.0f, 1.0f);
    size_t n = controlPoints.size();
    if (n <= 1) return Vector2D(0.0f, 0.0f);

    if (n == 4) {
        float u = 1.0f - t;
        float u2 = u * u;
        float t2 = t * t;

        Point d0 = controlPoints[1] - controlPoints[0];
        Point d1 = controlPoints[2] - controlPoints[1];
        Point d2 = controlPoints[3] - controlPoints[2];

        return d0 * (3.0f * u2) + d1 * (6.0f * u * t) + d2 * (3.0f * t2);
    }

    if (n == 2) {
        return controlPoints[1] - controlPoints[0];
    }

    // General Bézier derivative
    std::vector<Point> diffPoints(n - 1);
    float deg = static_cast<float>(n - 1);
    for (size_t i = 0; i < n - 1; ++i) {
        diffPoints[i] = (controlPoints[i + 1] - controlPoints[i]) * deg;
    }
    BezierCurve derivCurve(diffPoints);
    return derivCurve.getPoint(t);
}

Vector2D BezierCurve::getSecondDerivative(float t) const {
    t = std::clamp(t, 0.0f, 1.0f);
    size_t n = controlPoints.size();
    if (n <= 2) return Vector2D(0.0f, 0.0f);

    if (n == 4) {
        float u = 1.0f - t;
        Point p0 = controlPoints[0];
        Point p1 = controlPoints[1];
        Point p2 = controlPoints[2];
        Point p3 = controlPoints[3];

        Point a = (p2 - p1 * 2.0f + p0) * (6.0f * u);
        Point b = (p3 - p2 * 2.0f + p1) * (6.0f * t);
        return a + b;
    }

    // Numerical finite-difference approximation for arbitrary orders
    const float dt = 1e-3f;
    Vector2D d1 = getDerivative(std::clamp(t - dt, 0.0f, 1.0f));
    Vector2D d2 = getDerivative(std::clamp(t + dt, 0.0f, 1.0f));
    return (d2 - d1) / (2.0f * dt);
}

Vector2D BezierCurve::getTangent(float t) const {
    Vector2D deriv = getDerivative(t);
    return deriv.normalized();
}

Vector2D BezierCurve::getNormal(float t) const {
    Vector2D tangent = getTangent(t);
    // Perpendicular vector rotated 90 degrees CCW (-y, x)
    return Vector2D(-tangent.y, tangent.x);
}

float BezierCurve::getCurvature(float t) const {
    Vector2D d1 = getDerivative(t);
    Vector2D d2 = getSecondDerivative(t);

    float num = d1.x * d2.y - d1.y * d2.x;
    float denom = std::pow(d1.x * d1.x + d1.y * d1.y, 1.5f);

    if (denom < 1e-6f) return 0.0f;
    return num / denom;
}

float BezierCurve::getLength(int samples) const {
    if (samples < 2) samples = 2;
    float totalLen = 0.0f;
    Point prev = getPoint(0.0f);

    for (int i = 1; i <= samples; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(samples);
        Point curr = getPoint(t);
        totalLen += curr.distanceTo(prev);
        prev = curr;
    }
    return totalLen;
}

float BezierCurve::getRemainingDistance(float t, int samples) const {
    t = std::clamp(t, 0.0f, 1.0f);
    if (t >= 0.999f) return 0.0f;
    if (samples < 2) samples = 2;

    float remaining = 0.0f;
    Point prev = getPoint(t);

    for (int i = 1; i <= samples; ++i) {
        float nextT = t + (1.0f - t) * (static_cast<float>(i) / static_cast<float>(samples));
        Point curr = getPoint(nextT);
        remaining += curr.distanceTo(prev);
        prev = curr;
    }
    return remaining;
}

float BezierCurve::project(const Point& robotPose, float tGuess) const {
    if (controlPoints.empty()) return 0.0f;

    // Step 1: Coarse Grid Search across the curve
    const int numSamples = 20;
    float bestT = std::clamp(tGuess, 0.0f, 1.0f);
    float minDistSq = getPoint(bestT).distanceSqTo(robotPose);

    for (int i = 0; i <= numSamples; ++i) {
        float tSample = static_cast<float>(i) / static_cast<float>(numSamples);
        float distSq = getPoint(tSample).distanceSqTo(robotPose);
        if (distSq < minDistSq) {
            minDistSq = distSq;
            bestT = tSample;
        }
    }

    // Step 2: Newton-Raphson Optimization for exact closest point
    // Objective: minimize f(t) = 0.5 * ||P(t) - R||^2
    // f'(t)  = (P(t) - R) . P'(t) = 0
    // f''(t) = ||P'(t)||^2 + (P(t) - R) . P''(t)
    float t = bestT;
    for (int iter = 0; iter < 5; ++iter) {
        Point p = getPoint(t);
        Vector2D pPrime = getDerivative(t);
        Vector2D pDoublePrime = getSecondDerivative(t);
        Vector2D diff = p - robotPose;

        float fPrime = diff.dot(pPrime);
        float fDoublePrime = pPrime.dot(pPrime) + diff.dot(pDoublePrime);

        if (std::abs(fDoublePrime) < 1e-6f) break;

        float delta = fPrime / fDoublePrime;
        t = std::clamp(t - delta, 0.0f, 1.0f);

        if (std::abs(delta) < 1e-4f) break;
    }

    return t;
}

} // namespace pedro
