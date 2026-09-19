#pragma once
#include <math.h>

// Portable production control core: also compiled to simulator/control.wasm.
// SI units. Field +X right, +Y north, heading clockwise from +Y.
// Wheel order FL, BL, FR, BR. No PROS, clock, simulator truth or motor access here.
namespace nationals {
constexpr double pi=3.14159265358979323846;
inline double limit(double x,double lo,double hi){return x<lo?lo:x>hi?hi:x;}
inline double wrap(double a){return remainder(a,2*pi);}
struct Pose {double x=0,y=0,h=0;};
struct Reference {Pose pose; double vx=0,vy=0,w=0;};
struct Feedback {Pose pose; double vx=0,vy=0,w=0; double wheel[4]={}; bool valid=true;};
struct Config {
    double radius=0.1885853785424542; // (width + wheelbase)/(2 sqrt(2)), meters
    double maxWheelSpeed=0.864462; // nominal no-load rim speed, not chassis speed
    double maxVoltage=12, maxAcceleration=2.0;
    double kS=0.40,kV=13.88,kA=0.15,wheelKp=2.5,wheelKi=0.8;
    double wheelKd=0; // Reference implementation uses PI + feedforward (D=0).
    double qPosition=4,qHeading=3,rTranslation=0.5,rRotation=0.5;
};
struct Output {double volts[4]={},target[4]={}; bool valid=false;};
struct Mat {
    double a[3][3]={};
    static Mat diagonal(double x,double y,double z){Mat m;m.a[0][0]=x;m.a[1][1]=y;m.a[2][2]=z;return m;}
    Mat t()const{Mat m;for(int i=0;i<3;++i)for(int j=0;j<3;++j)m.a[i][j]=a[j][i];return m;}
    Mat operator+(const Mat& b)const{Mat m;for(int i=0;i<3;++i)for(int j=0;j<3;++j)m.a[i][j]=a[i][j]+b.a[i][j];return m;}
    Mat operator-(const Mat& b)const{Mat m;for(int i=0;i<3;++i)for(int j=0;j<3;++j)m.a[i][j]=a[i][j]-b.a[i][j];return m;}
    Mat operator*(const Mat& b)const{Mat m;for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)m.a[i][j]+=a[i][k]*b.a[k][j];return m;}
    bool inverse(Mat& out)const{
        double v[3][6]={};for(int i=0;i<3;++i){for(int j=0;j<3;++j)v[i][j]=a[i][j];v[i][i+3]=1;}
        for(int k=0;k<3;++k){
            int p=k;for(int i=k+1;i<3;++i)if(fabs(v[i][k])>fabs(v[p][k]))p=i;
            if(!isfinite(v[p][k])||fabs(v[p][k])<1e-12)return false;
            for(int j=0;j<6;++j){double x=v[p][j];v[p][j]=v[k][j];v[k][j]=x;}
            double d=v[k][k];for(int j=0;j<6;++j)v[k][j]/=d;
            for(int i=0;i<3;++i)if(i!=k){double f=v[i][k];for(int j=0;j<6;++j)v[i][j]-=f*v[k][j];}
        }
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)out.a[i][j]=v[i][j+3];return true;
    }
};
inline bool validPose(const Pose& p){return isfinite(p.x)&&isfinite(p.y)&&isfinite(p.h);}
inline bool validConfig(const Config& c){
    const double positive[]={c.radius,c.maxWheelSpeed,c.maxVoltage,c.maxAcceleration,c.qPosition,c.qHeading,c.rTranslation,c.rRotation};
    for(double v:positive)if(!isfinite(v)||v<=0)return false;
    const double gains[]={c.kS,c.kV,c.kA,c.wheelKp,c.wheelKi,c.wheelKd};
    for(double v:gains)if(!isfinite(v)||v<0)return false;
    return c.maxVoltage<=12;
}
inline void inverseX(double s,double f,double w,double radius,double wheel[4]){
    constexpr double k=0.7071067811865475;
    wheel[0]=(f+s)*k+radius*w;wheel[1]=(f-s)*k+radius*w;
    wheel[2]=(f-s)*k-radius*w;wheel[3]=(f+s)*k-radius*w;
}
inline void forwardX(const double wheel[4],double radius,double& s,double& f,double& w){
    constexpr double k=0.3535533905932738;
    s=(wheel[0]-wheel[1]-wheel[2]+wheel[3])*k;
    f=(wheel[0]+wheel[1]+wheel[2]+wheel[3])*k;
    w=(wheel[0]+wheel[1]-wheel[2]-wheel[3])/(4*radius);
}
class WheelVelocity {
    double integral[4]={},previous[4]={},previousMeasured[4]={},derivative[4]={}; bool first=true;
public:
    void reset(){for(int i=0;i<4;++i)integral[i]=previous[i]=previousMeasured[i]=derivative[i]=0;first=true;}
    Output update(double s,double f,double w,const Feedback& fb,double dt,const Config& c){
        Output o;
        if(!fb.valid||!isfinite(dt)||dt<=0||dt>0.1||!validConfig(c)||!isfinite(s)||!isfinite(f)||!isfinite(w)){reset();return o;}
        for(double v:fb.wheel)if(!isfinite(v)){reset();return o;}
        inverseX(s,f,w,c.radius,o.target);
        double peak=1;for(double v:o.target)peak=fmax(peak,fabs(v)/c.maxWheelSpeed);
        // Uniform wheel desaturation preserves the requested twist direction.
        double ramp=1;
        for(int i=0;i<4;++i){o.target[i]/=peak;ramp=fmax(ramp,fabs(o.target[i]-previous[i])/(c.maxAcceleration*dt));}
        for(int i=0;i<4;++i){
            o.target[i]=previous[i]+(o.target[i]-previous[i])/ramp;
            const double acceleration=(o.target[i]-previous[i])/dt;
            const double error=o.target[i]-fb.wheel[i];
            const double candidate=limit(integral[i]+error*dt,-2,2);
            double raw=first?0:(fb.wheel[i]-previousMeasured[i])/dt;
            derivative[i]+=(1-exp(-dt/0.03))*(raw-derivative[i]);previousMeasured[i]=fb.wheel[i];
            const double ff=c.kS*tanh(o.target[i]/0.04)+c.kV*o.target[i]+c.kA*acceleration;
            double voltage=ff+c.wheelKp*error+c.wheelKi*candidate-c.wheelKd*derivative[i];
            // Integrate only when unsaturated or when error unwinds saturation.
            if(fabs(voltage)<=c.maxVoltage||voltage*error<0)integral[i]=candidate;
            voltage=ff+c.wheelKp*error+c.wheelKi*integral[i]-c.wheelKd*derivative[i];
            o.volts[i]=limit(voltage,-c.maxVoltage,c.maxVoltage);previous[i]=o.target[i];
        }
        first=false;o.valid=true;return o;
    }
};
class Cascade {
    Mat gain;bool gainValid=false;double lastH=0,lastVx=0,lastVy=0,lastDt=0;
    bool solve(const Reference& r,double dt,const Config& c){
        // Linearize field kinematics about the reference body velocity/heading.
        // x_dot = s cos(h)+f sin(h); y_dot=-s sin(h)+f cos(h); h_dot=w.
        Mat A=Mat::diagonal(1,1,1),B;
        A.a[0][2]=r.vy*dt;A.a[1][2]=-r.vx*dt;
        B.a[0][0]=cos(r.pose.h)*dt;B.a[0][1]=sin(r.pose.h)*dt;
        B.a[1][0]=-sin(r.pose.h)*dt;B.a[1][1]=cos(r.pose.h)*dt;B.a[2][2]=dt;
        const Mat Q=Mat::diagonal(c.qPosition,c.qPosition,c.qHeading);
        const Mat R=Mat::diagonal(c.rTranslation,c.rTranslation,c.rRotation);
        Mat P=Q,K;bool converged=false;
        for(int it=0;it<600;++it){
            Mat inv;if(!(R+B.t()*P*B).inverse(inv))return false;
            K=inv*B.t()*P*A;
            Mat next=Q+A.t()*P*A-A.t()*P*B*K;
            double delta=0,scale=1;
            for(int i=0;i<3;++i)for(int j=0;j<3;++j){if(!isfinite(next.a[i][j]))return false;delta=fmax(delta,fabs(next.a[i][j]-P.a[i][j]));scale=fmax(scale,fabs(next.a[i][j]));}
            P=next;if(delta<1e-7*scale){converged=true;break;}
        }
        if(!converged)return false;
        Mat inv;if(!(R+B.t()*P*B).inverse(inv))return false;
        gain=inv*B.t()*P*A;gainValid=true;lastH=r.pose.h;lastVx=r.vx;lastVy=r.vy;lastDt=dt;return true;
    }
public:
    Config config;WheelVelocity wheels;
    void reset(){gainValid=false;wheels.reset();}
    Output update(const Reference& r,const Feedback& fb,double dt){
        if(!fb.valid||!validPose(fb.pose)||!validPose(r.pose)||!isfinite(dt)||dt<=0||dt>0.1||!validConfig(config)){reset();return {};}
        const double signals[]={r.vx,r.vy,r.w,fb.vx,fb.vy,fb.w};for(double v:signals)if(!isfinite(v)){reset();return {};}
        double error[3]={r.pose.x-fb.pose.x,r.pose.y-fb.pose.y,wrap(r.pose.h-fb.pose.h)};
        double s=0,f=0,w=r.w;
        if(!gainValid||fabs(wrap(r.pose.h-lastH))>0.02||fabs(r.vx-lastVx)>0.02||fabs(r.vy-lastVy)>0.02||fabs(dt-lastDt)>0.0005)
            if(!solve(r,dt,config)){reset();return {};}
        double u[3]={};for(int i=0;i<3;++i)for(int j=0;j<3;++j)u[i]+=gain.a[i][j]*error[j];
        s=r.vx*cos(r.pose.h)-r.vy*sin(r.pose.h)+u[0];
        f=r.vx*sin(r.pose.h)+r.vy*cos(r.pose.h)+u[1];w+=u[2];
        return wheels.update(s,f,w,fb,dt,config);
    }
};
// Smooth holonomic point/heading reference: heading is independent of translation.
// Quintic time law bounds speed, acceleration and jerk analytically.
inline double duration(const Pose& a,const Pose& b,double speed=0.45,double accel=0.8,double jerk=3.5){
    if(!validPose(a)||!validPose(b)||!isfinite(speed)||!isfinite(accel)||!isfinite(jerk)||speed<=0||accel<=0||jerk<=0)return -1;
    double d=hypot(b.x-a.x,b.y-a.y),h=fabs(wrap(b.h-a.h));
    return fmax(0.25,fmax(fmax(1.875*d/speed,sqrt(5.774*d/accel)),fmax(cbrt(60*d/jerk),1.875*h/1.5)));
}
inline Reference pointReference(const Pose& a,const Pose& b,double time,double T){
    double u=limit(time/T,0,1),u2=u*u,u3=u2*u,u4=u3*u,u5=u4*u;
    double q=10*u3-15*u4+6*u5,dq=(30*u2-60*u3+30*u4)/T;
    double dh=wrap(b.h-a.h);
    return {{a.x+(b.x-a.x)*q,a.y+(b.y-a.y)*q,a.h+dh*q},(b.x-a.x)*dq,(b.y-a.y)*dq,dh*dq};
}
inline bool settled(const Reference& r,const Feedback& f){return hypot(r.pose.x-f.pose.x,r.pose.y-f.pose.y)<0.02032&&fabs(wrap(r.pose.h-f.pose.h))<0.035&&hypot(f.vx,f.vy)<0.0254&&fabs(f.w)<0.0873;}
} // namespace nationals
