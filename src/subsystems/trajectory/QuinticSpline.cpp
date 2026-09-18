#include "subsystems/trajectory/QuinticSpline.hpp"
#include "lemlib/util.hpp"
#include <cmath>
#include <algorithm>
#include <array>

namespace lemlib {

std::vector<State> QuinticSplineGenerator::generateTrajectory(const SplineWaypoints& params, double dt) {
    const double values[]={dt,params.start.x,params.start.y,params.start.theta,params.end.x,params.end.y,params.end.theta,params.maxVel,params.maxAccel,params.maxJerk,params.startVel,params.endVel};
    for(double v:values) if(!std::isfinite(v)) return {};
    if(dt<0.001||dt>0.1||params.maxVel<=0||params.maxAccel<=0||params.maxJerk<=0||params.startVel<0||params.endVel<0||params.startVel>params.maxVel||params.endVel>params.maxVel) return {};

    // Convert start and end poses from inches to meters
    double x0 = params.start.x * INCH_TO_METER;
    double y0 = params.start.y * INCH_TO_METER;
    double theta0 = degToRad(params.start.theta);

    double x1 = params.end.x * INCH_TO_METER;
    double y1 = params.end.y * INCH_TO_METER;
    double theta1 = degToRad(params.end.theta);

    double dist = std::hypot(x1 - x0, y1 - y0);
    if (dist < 1e-3) {
        return { State{x0, y0, M_PI_2-theta0, 0.0, 0.0,0.0} };
    }

    // Tangent vectors scaled by distance to create a pronounced, smooth curvature arc
    double scale = std::max(1.35 * dist, 0.45);
    double vx0 = scale * std::sin(theta0); // In LemLib heading, dx = sin(theta), dy = cos(theta)
    double vy0 = scale * std::cos(theta0);
    double vx1 = scale * std::sin(theta1);
    double vy1 = scale * std::cos(theta1);

    // Initial and final accelerations (0 for smooth start/stop)
    double ax0 = 0.0, ay0 = 0.0;
    double ax1 = 0.0, ay1 = 0.0;

    // Quintic polynomial coefficients for X(s) where s in [0, 1]
    double cx0 = x0;
    double cx1 = vx0;
    double cx2 = 0.5 * ax0;
    double cx3 = 10.0 * (x1 - x0) - (6.0 * vx0 + 4.0 * vx1) - (1.5 * ax0 - 0.5 * ax1);
    double cx4 = -15.0 * (x1 - x0) + (8.0 * vx0 + 7.0 * vx1) + (1.5 * ax0 - ax1);
    double cx5 = 6.0 * (x1 - x0) - 3.0 * (vx0 + vx1) - 0.5 * (ax0 - ax1);

    // Quintic polynomial coefficients for Y(s) where s in [0, 1]
    double cy0 = y0;
    double cy1 = vy0;
    double cy2 = 0.5 * ay0;
    double cy3 = 10.0 * (y1 - y0) - (6.0 * vy0 + 4.0 * vy1) - (1.5 * ay0 - 0.5 * ay1);
    double cy4 = -15.0 * (y1 - y0) + (8.0 * vy0 + 7.0 * vy1) + (1.5 * ay0 - ay1);
    double cy5 = 6.0 * (y1 - y0) - 3.0 * (vy0 + vy1) - 0.5 * (ay0 - ay1);

    auto geometry=[&](double u) {
        const double u2=u*u,u3=u2*u,u4=u3*u,u5=u4*u;
        return std::array<double,6>{
            cx0+cx1*u+cx2*u2+cx3*u3+cx4*u4+cx5*u5,
            cy0+cy1*u+cy2*u2+cy3*u3+cy4*u4+cy5*u5,
            cx1+2*cx2*u+3*cx3*u2+4*cx4*u3+5*cx5*u4,
            cy1+2*cy2*u+3*cy3*u2+4*cy4*u3+5*cy5*u4,
            2*cx2+6*cx3*u+12*cx4*u2+20*cx5*u3,
            2*cy2+6*cy3*u+12*cy4*u2+20*cy5*u3};
    };
    // Arc-length table decouples spline geometry from its time law.
    constexpr int resolution=2048;
    std::array<double,resolution+1> arc{};
    auto prev=geometry(0);
    for(int i=1;i<=resolution;++i) {
        auto p=geometry(double(i)/resolution);arc[i]=arc[i-1]+std::hypot(p[0]-prev[0],p[1]-prev[1]);prev=p;
    }
    const double length=arc.back();
    double duration=std::max(length/params.maxVel,dt);
    std::array<double,6> c{};
    bool feasible=false;
    // Quintic distance(t), with specified endpoint speeds and zero endpoint acceleration.
    // Limits are tangential (curvature acceleration requires an additional chassis limit).
    for(int attempt=0;attempt<1200;++attempt) {
        const double v0=params.startVel*duration,v1=params.endVel*duration;
        c={0,v0,0,10*length-6*v0-4*v1,-15*length+8*v0+7*v1,6*length-3*v0-3*v1};
        feasible=true;
        for(int i=0;i<=2000;++i) {
            const double u=double(i)/2000,u2=u*u,u3=u2*u,u4=u3*u;
            const double v=(c[1]+3*c[3]*u2+4*c[4]*u3+5*c[5]*u4)/duration;
            const double a=(6*c[3]*u+12*c[4]*u2+20*c[5]*u3)/(duration*duration);
            const double j=(6*c[3]+24*c[4]*u+60*c[5]*u2)/(duration*duration*duration);
            if(v < -1e-8 || v>params.maxVel*0.999999 || std::abs(a)>params.maxAccel*0.999999 || std::abs(j)>params.maxJerk*0.999999) {feasible=false;break;}
        }
        if(feasible) break;
        duration*=1.01;
        if(duration>120 || duration/dt>120000) return {};
    }
    if(!feasible) return {};
    const int steps=std::ceil(duration/dt);
    std::vector<State> trajectory;trajectory.reserve(steps+1);
    for(int i=0;i<=steps;++i) {
        const double time=std::min(i*dt,duration),u=time/duration,u2=u*u,u3=u2*u,u4=u3*u,u5=u4*u;
        const double distance=std::clamp(c[1]*u+c[3]*u3+c[4]*u4+c[5]*u5,0.0,length);
        const double v=(c[1]+3*c[3]*u2+4*c[4]*u3+5*c[5]*u4)/duration;
        auto at=std::lower_bound(arc.begin(),arc.end(),distance);
        int k=std::clamp(int(at-arc.begin()),1,resolution);
        const double span=arc[k]-arc[k-1];
        const double parameter=(k-1+(span>1e-12?(distance-arc[k-1])/span:0))/resolution;
        auto p=geometry(parameter);const double norm=std::hypot(p[2],p[3]);
        if(norm<1e-8) return {}; // A cusp needs an explicit stop/turn, not an undefined tangent.
        const double curvature=(p[2]*p[5]-p[3]*p[4])/(norm*norm*norm);
        trajectory.push_back({p[0],p[1],std::atan2(p[3],p[2]),v,curvature*v,time});
    }

    return trajectory;
}

std::vector<State> QuinticSplineGenerator::generateMultiPointTrajectory(
    const std::vector<Pose>& waypoints, double maxVel, double maxAccel, double dt) {
    if (waypoints.size() < 2) return {};

    std::vector<State> fullTrajectory;

    for (size_t i = 0; i < waypoints.size() - 1; ++i) {
        SplineWaypoints segment;
        segment.start = waypoints[i];
        segment.end = waypoints[i + 1];
        segment.maxVel = maxVel;
        segment.maxAccel = maxAccel;

        auto segTrajectory = generateTrajectory(segment, dt);
        if(segTrajectory.empty()) return {};
        const double offset=fullTrajectory.empty()?0:fullTrajectory.back().time;
        for(auto& state:segTrajectory) state.time+=offset;
        if (i > 0 && !segTrajectory.empty()) {
            segTrajectory.erase(segTrajectory.begin()); // prevent duplicate connection points
        }
        fullTrajectory.insert(fullTrajectory.end(), segTrajectory.begin(), segTrajectory.end());
    }

    return fullTrajectory;
}

} // namespace lemlib
