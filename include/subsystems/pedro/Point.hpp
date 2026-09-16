#pragma once

#include <cmath>

namespace pedro {

/**
 * @brief 2D Point and Vector math structure for Pedro Pathing
 */
struct Point {
    float x = 0.0f;
    float y = 0.0f;

    Point() = default;
    Point(float x, float y) : x(x), y(y) {}

    Point operator+(const Point& other) const { return Point(x + other.x, y + other.y); }
    Point operator-(const Point& other) const { return Point(x - other.x, y - other.y); }
    Point operator*(float scalar) const { return Point(x * scalar, y * scalar); }
    Point operator/(float scalar) const { return Point(x / scalar, y / scalar); }

    Point& operator+=(const Point& other) { x += other.x; y += other.y; return *this; }
    Point& operator-=(const Point& other) { x -= other.x; y -= other.y; return *this; }
    Point& operator*=(float scalar) { x *= scalar; y *= scalar; return *this; }

    float dot(const Point& other) const { return x * other.x + y * other.y; }
    float cross(const Point& other) const { return x * other.y - y * other.x; }

    float distanceTo(const Point& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    float distanceSqTo(const Point& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        return dx * dx + dy * dy;
    }

    float magnitude() const { return std::sqrt(x * x + y * y); }
    float magnitudeSq() const { return x * x + y * y; }

    Point normalized() const {
        float mag = magnitude();
        if (mag < 1e-6f) return Point(0.0f, 0.0f);
        return Point(x / mag, y / mag);
    }

    Point rotate(float angleRad) const {
        float c = std::cos(angleRad);
        float s = std::sin(angleRad);
        return Point(x * c - y * s, x * s + y * c);
    }
};

using Vector2D = Point;

} // namespace pedro
