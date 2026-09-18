#include "subsystems/ekf/EKF.hpp"
#include "lemlib/safety.hpp"
#include "lemlib/util.hpp"
#include <cmath>
namespace lemlib {
namespace {
template<int M> bool correct(Eigen::Matrix<double,5,1>& x,Eigen::Matrix<double,5,5>& p,
    const Eigen::Matrix<double,M,5>& h,const Eigen::Matrix<double,M,1>& z,
    const Eigen::Matrix<double,M,M>& r,int angular=-1) {
    Eigen::Matrix<double,M,1> residual=z-h*x;
    if(angular>=0) residual(angular)=std::remainder(residual(angular),2*M_PI);
    Eigen::Matrix<double,M,M> s=h*p*h.transpose()+r;
    auto solve=s.ldlt();
    if(solve.info()!=Eigen::Success || !solve.isPositive()) return false;
    const auto normalized=solve.solve(residual).eval();
    if(!normalized.allFinite() || residual.dot(normalized)>25.0+M) return false;
    Eigen::Matrix<double,5,M> k=solve.solve((p*h.transpose()).transpose()).transpose();
    auto candidate=(x+k*residual).eval();
    candidate(2)=std::remainder(candidate(2),2*M_PI);
    const Eigen::Matrix<double,5,5> a=Eigen::Matrix<double,5,5>::Identity()-k*h;
    Eigen::Matrix<double,5,5> covariance=a*p*a.transpose()+k*r*k.transpose();
    covariance=(0.5*(covariance+covariance.transpose())).eval();
    if(!candidate.allFinite()||!covariance.allFinite()||!covariance.ldlt().isPositive()) return false;
    x=candidate;p=covariance;return true;
}
bool positive(double v){return std::isfinite(v)&&v>0;}
}
double RobotEKF::normalizeAngle(double angle) {return std::isfinite(angle)?std::remainder(angle,2*M_PI):0;}
RobotEKF::RobotEKF(const Pose& p) {reset(Pose(0,0,0));reset(p);}
void RobotEKF::reset(const Pose& p) {
    if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.theta)) return;
    Lock guard(ekfMutex);
    x_est.setZero(); x_est(0)=p.x*0.0254; x_est(1)=p.y*0.0254; x_est(2)=normalizeAngle(M_PI_2-degToRad(p.theta));
    P_cov.setZero(); P_cov.diagonal()<<0.01,0.01,0.005,0.1,0.1;
    // Spectral densities per second; equivalent to old nominal 100Hz Q.
    Q_noise.setZero(); Q_noise.diagonal()<<0.01,0.01,0.005,1,1;
}
void RobotEKF::predict(double dt) {
    if(!positive(dt)||dt>0.1) return;
    Lock guard(ekfMutex);
    const double t=x_est(2),v=x_est(3),w=x_est(4);
    Eigen::Matrix<double,5,5> f=Eigen::Matrix<double,5,5>::Identity();
    f(0,2)=-v*std::sin(t)*dt;f(0,3)=std::cos(t)*dt;
    f(1,2)=v*std::cos(t)*dt;f(1,3)=std::sin(t)*dt;f(2,4)=dt;
    x_est(0)+=v*std::cos(t)*dt;x_est(1)+=v*std::sin(t)*dt;x_est(2)=normalizeAngle(t+w*dt);
    P_cov=(f*P_cov*f.transpose()+Q_noise*dt).eval();
    P_cov=(0.5*(P_cov+P_cov.transpose())).eval();
}
void RobotEKF::updatePose(double x,double y,double heading,double sp,double sh) {
    if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(heading)||!positive(sp)||!positive(sh)) return;
    Lock guard(ekfMutex);
    Eigen::Matrix<double,3,5> h=Eigen::Matrix<double,3,5>::Zero();h(0,0)=h(1,1)=h(2,2)=1;
    Eigen::Vector3d z(x,y,normalizeAngle(M_PI_2-heading));
    Eigen::Matrix3d r=Eigen::Matrix3d::Zero();r.diagonal()<<sp*sp,sp*sp,sh*sh;
    correct<3>(x_est,P_cov,h,z,r,2);
}
void RobotEKF::updateIMU(double clockwiseRate,double stddev) {
    if(!std::isfinite(clockwiseRate)||!positive(stddev)) return;
    Lock guard(ekfMutex);
    Eigen::Matrix<double,1,5> h=Eigen::Matrix<double,1,5>::Zero();h(0,4)=1;
    Eigen::Matrix<double,1,1> z,r;z(0)=-clockwiseRate;r(0)=stddev*stddev;
    correct<1>(x_est,P_cov,h,z,r);
}
void RobotEKF::updateWheelVelocities(double left,double right,double width,double stddev) {
    if(!std::isfinite(left)||!std::isfinite(right)||!positive(width)||width<1e-4||!positive(stddev)) return;
    Lock guard(ekfMutex);
    Eigen::Matrix<double,2,5> h=Eigen::Matrix<double,2,5>::Zero();h(0,3)=h(1,4)=1;
    Eigen::Vector2d z((left+right)/2,(right-left)/width);
    Eigen::Matrix2d r=Eigen::Matrix2d::Zero();r.diagonal()<<0.5*stddev*stddev,2*stddev*stddev/(width*width);
    correct<2>(x_est,P_cov,h,z,r);
}
Pose RobotEKF::getPose() const {Lock guard(ekfMutex);return {float(x_est(0)/0.0254),float(x_est(1)/0.0254),float(radToDeg(M_PI_2-x_est(2)))};}
double RobotEKF::getLinearVelocity() const {Lock guard(ekfMutex);return x_est(3);}
double RobotEKF::getAngularVelocity() const {Lock guard(ekfMutex);return -x_est(4);}
Eigen::Matrix<double,5,5> RobotEKF::getCovariance() const {Lock guard(ekfMutex);return P_cov;}
}
