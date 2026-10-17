#pragma once
#include "Cascade.hpp"
#include "subsystems/pedro/BezierCurve.hpp"
#include <array>

namespace nationals {
struct FieldReference {
    Reference reference{{NAN,NAN,NAN}};
    double parameter=0,remaining=0,crossTrack=0;
    bool valid=false;
};

class GuidingVectorField {
    pedro::BezierCurve curve;
    std::array<double,513> arc{};
    double startHeading=0,deltaHeading=0,length=0;
    Pose endpoint;
    bool available=false;

    pedro::Point direction(double parameter)const{
        auto derivative=curve.getDerivative(parameter);
        double magnitude=hypot(derivative.x,derivative.y);
        for(double span=1.0/512;magnitude<1e-6&&span<=1;span*=2){
            derivative=curve.getPoint(fmin(1,parameter+span))-curve.getPoint(fmax(0,parameter-span));
            magnitude=hypot(derivative.x,derivative.y);
        }
        return magnitude>1e-12?pedro::Point(float(derivative.x/magnitude),float(derivative.y/magnitude)):pedro::Point{};
    }
public:
    static constexpr int contractVersion=2;
    static constexpr double maxAngularSpeed=1.5;
    static constexpr double headingPrecision=0.005;
    GuidingVectorField(const pedro::BezierCurve& path,double start,double end):curve(path),startHeading(start),deltaHeading(wrap(end-start)){
        const auto& points=curve.getControlPoints();
        if((points.size()!=2&&points.size()!=4)||!isfinite(start)||!isfinite(end))return;
        for(const auto& point:points)if(!isfinite(point.x)||!isfinite(point.y)||fabs(point.x)>60||fabs(point.y)>60)return;
        const auto chord=points.back()-points.front();
        const double chordSquared=chord.magnitudeSq();
        for(size_t index=1;index<points.size();++index){
            const auto edge=points[index]-points[index-1];
            if(chordSquared<1e-10){if(edge.magnitudeSq()>1e-10)return;}
            else if(edge.dot(chord)<0)return;
        }
        auto previous=curve.getPoint(0);
        for(int index=1;index<=512;++index){
            auto point=curve.getPoint(float(index)/512);
            arc[index]=arc[index-1]+previous.distanceTo(point)*0.0254;
            previous=point;
        }
        length=arc.back();
        if(!isfinite(length))return;
        const auto goal=curve.getEndPoint();
        endpoint={goal.x*0.0254,goal.y*0.0254,end};
        available=true;
    }
    bool valid()const{return available;}
    Reference goal()const{return {endpoint};}
    FieldReference at(const Feedback& feedback)const{
        FieldReference result;
        if(!available||!feedback.valid||!validPose(feedback.pose)||!isfinite(feedback.vx)||!isfinite(feedback.vy)||!isfinite(feedback.w))return result;
        const bool atEndpoint=hypot(feedback.pose.x-endpoint.x,feedback.pose.y-endpoint.y)<1e-6;
        const double parameter=length<1e-6||atEndpoint?1:curve.project({float(feedback.pose.x/0.0254),float(feedback.pose.y/0.0254)});
        const auto point=curve.getPoint(parameter);
        const double scaled=limit(parameter*512,0,512);
        const int index=std::min(int(scaled),511);
        const double travelled=arc[index]+(arc[index+1]-arc[index])*(scaled-index);
        result.parameter=parameter;
        result.remaining=fmax(0,length-travelled);
        const double errorX=point.x*0.0254-feedback.pose.x,errorY=point.y*0.0254-feedback.pose.y;
        result.crossTrack=hypot(errorX,errorY);
        const double progress=length<1e-6?1:travelled/length;
        const auto tangent=direction(parameter);
        const double curvature=fabs(curve.getCurvature(parameter))/0.0254;
        double speed=fmin(0.35,sqrt(2*0.6*result.remaining));
        if(curvature>1e-9)speed=fmin(speed,sqrt(0.6/curvature));
        double velocityX=tangent.x*speed+1.5*errorX,velocityY=tangent.y*speed+1.5*errorY;
        const double scale=fmax(1,hypot(velocityX,velocityY)/0.35);
        velocityX/=scale;velocityY/=scale;
        const double along=limit(feedback.vx*tangent.x+feedback.vy*tangent.y,0,0.35);
        const double yaw=length<1e-6||parameter>=1?0:limit(deltaHeading*along/length,-maxAngularSpeed,maxAngularSpeed);
        result.reference={{point.x*0.0254,point.y*0.0254,startHeading+deltaHeading*progress},velocityX,velocityY,yaw};
        result.valid=validPose(result.reference.pose)&&isfinite(velocityX)&&isfinite(velocityY)&&isfinite(yaw)&&isfinite(result.remaining)&&isfinite(result.crossTrack);
        return result;
    }
    bool complete(const Feedback& feedback)const{
        const auto sample=at(feedback);
        return sample.valid&&sample.remaining<0.02032&&settled(goal(),feedback)&&fabs(wrap(endpoint.h-feedback.pose.h))<headingPrecision;
    }
};
}
