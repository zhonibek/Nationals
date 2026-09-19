#pragma once
#include "Cascade.hpp"
#include "subsystems/pedro/BezierCurve.hpp"
#include <array>

namespace nationals {
// Pedro Bezier geometry feeds the same LTV-LQR -> wheel PID cascade.
// Heading is independent of the tangent. The legacy GVF follower remains
// available in the library, but is not an alternative motor writer in main.
class BezierReference {
    pedro::BezierCurve curve;
    std::array<double,513> arc{};
    double startHeading=0,deltaHeading=0,length=0;
public:
    double seconds=-1;
    BezierReference(const pedro::BezierCurve& curve,double start,double end):curve(curve),startHeading(start),deltaHeading(wrap(end-start)){
        if(curve.getControlPoints().size()<2||!isfinite(start)||!isfinite(end))return;
        for(const auto& p:curve.getControlPoints())if(!isfinite(p.x)||!isfinite(p.y))return;
        auto prev=curve.getPoint(0);
        for(int i=1;i<=512;i++){auto p=curve.getPoint(float(i)/512);arc[i]=arc[i-1]+prev.distanceTo(p)*0.0254;prev=p;}
        length=arc.back();if(length<1e-6)return;
        seconds=duration({0,0,start},{length,0,end},0.35,0.6,2.5);
        // Bound centripetal acceleration and heading rate by slowing the time law.
        double peak=1;
        for(int i=0;i<=512;i++){
            double curvature=fabs(curve.getCurvature(float(i)/512))/0.0254;
            peak=fmax(peak,sqrt(curvature*0.35*0.35/0.6));
        }
        seconds*=peak;
    }
    Reference at(double time)const{
        if(seconds<=0)return {{NAN,NAN,NAN}};
        const auto profile=pointReference({0,0,startHeading},{length,0,startHeading+deltaHeading},time,seconds);
        auto it=std::lower_bound(arc.begin(),arc.end(),profile.pose.x);
        int i=std::clamp(int(it-arc.begin()),1,512);double span=arc[i]-arc[i-1];
        double t=(i-1+(span>1e-12?(profile.pose.x-arc[i-1])/span:0))/512;
        auto p=curve.getPoint(t),d=curve.getTangent(t);
        return {{p.x*0.0254,p.y*0.0254,profile.pose.h},d.x*profile.vx,d.y*profile.vx,profile.w};
    }
};
}
