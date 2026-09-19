#include "subsystems/ltv/ltv.hpp"
#include "lemlib/util.hpp"
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <new>
#include "lemlib/chassis/odom.hpp"
#include "subsystems/control/HolonomicMotion.hpp"

namespace lemlib {

std::string Vector2::latex() const {
    std::ostringstream oss;
    oss << "\\left(" << std::fixed << std::setprecision(3) << this->x << "," << this->y << "\\right)";
    return oss.str();
}

double LTVPathFollower::angleError(double robotAngle, double targetAngle) {
    return std::remainder(targetAngle - robotAngle, 2.0 * M_PI);
}

double LTVPathFollower::clamp(double value, double min, double max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

LTVPathFollower::LTVPathFollower(Chassis& chassis, pros::MotorGroup& leftMotors, pros::MotorGroup& rightMotors,
                                 const VelocityControllerConfig& config)
    : chassis(chassis),
      leftMotors(leftMotors),
      rightMotors(rightMotors),
      controller(config), velocityConfig(config) {
    // 3.25" wheel: circumference = pi * 3.25 * 0.0254 = 0.259339 m
    double wheelDiameterMeters = config.wheelDiameterMeters*config.externalGearRatio;
    rpm_to_mps_factor = (M_PI * wheelDiameterMeters) / 60.0f;
}

LTVPathFollower::~LTVPathFollower(){cancel();if(task){task->join();delete task;}}

void LTVPathFollower::followPath(const std::string& path_name, const ltvConfig& l_config) {
    Lock lifecycle(lifecycleMutex);
    if (is_running) {
        cancel();
        waitUntilDone();
    }

    if(task) {task->join();delete task;task=nullptr;}
    result=MotionResult::Running;
    is_running = true;
    cancel_request = false;
    distance_traveled_inches = 0.0f;

    TaskParams* params = new (std::nothrow) TaskParams{this, path_name, l_config, {}};
    task = params ? new (std::nothrow) pros::Task(task_trampoline, params, "LTVTask") : nullptr;
    
    if (task == nullptr || static_cast<pros::task_t>(*task) == nullptr) {
        delete params;
        delete task; task=nullptr;
        result=MotionResult::Busy;
        is_running = false;
        std::cout << "[LTV] Failed to start task!" << std::endl;
        return;
    }
    pros::delay(10);
}

void LTVPathFollower::followTrajectory(const std::vector<State>& trajectory, const ltvConfig& l_config) {
    Lock lifecycle(lifecycleMutex);
    if (is_running) {
        cancel();
        waitUntilDone();
    }

    if(task) {task->join();delete task;task=nullptr;}
    result=MotionResult::Running;
    is_running = true;
    cancel_request = false;
    distance_traveled_inches = 0.0f;

    TaskParams* params = new (std::nothrow) TaskParams{this, "", l_config, trajectory};
    task = params ? new (std::nothrow) pros::Task(task_trampoline, params, "LTVTask") : nullptr;
    
    if (task == nullptr || static_cast<pros::task_t>(*task) == nullptr) {
        delete params;
        delete task; task=nullptr;
        result=MotionResult::Busy;
        is_running = false;
        std::cout << "[LTV] Failed to start task!" << std::endl;
        return;
    }
    pros::delay(10);
}

double LTVPathFollower::getPathLength(const std::string& path_name) {
    std::vector<State> trajectory = prepare_trajectory(path_name);
    if (trajectory.empty()) return 0.0;
    double length_meters = 0.0;
    for (size_t i = 1; i < trajectory.size(); ++i) {
        double dx = trajectory[i].x - trajectory[i-1].x;
        double dy = trajectory[i].y - trajectory[i-1].y;
        length_meters += std::sqrt(dx*dx + dy*dy);
    }
    return length_meters * METER_TO_INCH;
}

void LTVPathFollower::task_trampoline(void* params) {
    TaskParams* p = static_cast<TaskParams*>(params);
    if (p && p->instance) {
        try {p->instance->followPathImpl(p->path_name, p->config, p->dynamic_path);}
        catch(...) {p->instance->result=MotionResult::InvalidInput;p->instance->is_running=false;}
    }
    delete p;
}

void LTVPathFollower::waitUntilDone() {
    while (is_running) {
        pros::delay(10);
    }
}

void LTVPathFollower::waitUntil(float dist_inches) {
    while (is_running && distance_traveled_inches < dist_inches) {
        pros::delay(10);
    }
}

void LTVPathFollower::waitUntil(float x_inch, float y_inch, float radius_inch) {
    while (is_running) {
        lemlib::Pose p = chassis.getPose();
        float dist = std::sqrt(std::pow(p.x - x_inch, 2) + std::pow(p.y - y_inch, 2));
        if (dist < radius_inch) {
            break;
        }
        pros::delay(10);
    }
}

void LTVPathFollower::cancel() {
    cancel_request = true;
}

bool LTVPathFollower::isRunning() {
    return is_running;
}

void LTVPathFollower::followPathImpl(const std::string& path_name, const ltvConfig& l_config, const std::vector<State>& dynamic_path) {
    std::vector<State> trajectory;
    struct Completion {std::atomic<bool>& running;~Completion(){running=false;}} completion{is_running};

    if (abortAuton) {
        result=MotionResult::Cancelled;
        is_running = false;
        return;
    }

    if (!dynamic_path.empty()) {
        trajectory = dynamic_path;
    } else {Lock guard(cacheMutex);auto it=precomputed_paths.find(path_name);if(it!=precomputed_paths.end()) trajectory=it->second;}

    if (trajectory.empty() && !path_name.empty()) {
        trajectory = prepare_trajectory(path_name);
    }

    if (trajectory.empty()) {
        result=MotionResult::InvalidInput;
        is_running = false;
        return;
    }

    if(chassis.getDrivebaseType()==DrivebaseType::MECANUM || !std::isfinite(rpm_to_mps_factor)||rpm_to_mps_factor<=0) {result=MotionResult::InvalidInput;return;}
    for(size_t i=0;i<trajectory.size();++i) {
        const auto& t=trajectory[i];const double values[]={t.x,t.y,t.heading,t.linear_vel,t.angular_vel,t.time};
        for(double v:values) if(!std::isfinite(v)) {result=MotionResult::InvalidInput;return;}
        if(t.time<0 || (i>0&&t.time<=trajectory[i-1].time)) {result=MotionResult::InvalidInput;return;}
    }
    if(trajectory.back().time>120 || l_config.settleTimeout<=0 || l_config.settleTimeout>15) {result=MotionResult::InvalidInput;return;}
    const float weights[]={l_config.q_x,l_config.q_y,l_config.q_theta,l_config.q_x_b,l_config.q_y_b,l_config.q_theta_b,l_config.r_vel,l_config.r_ang,l_config.r_vel_b,l_config.r_ang_b,l_config.q_scalar,l_config.max_lin_correction,l_config.max_ang_correction};
    for(float value:weights) if(!std::isfinite(value)||value<=0) {result=MotionResult::InvalidInput;return;}
    if (l_config.test) {
        double start_theta = l_config.backwards ? trajectory[0].heading + M_PI : trajectory[0].heading;
        double gps_start_theta = M_PI_2 - start_theta;
        chassis.setPose(trajectory[0].x / INCH_TO_METER, trajectory[0].y / INCH_TO_METER, lemlib::radToDeg(gps_start_theta));
    } else if (l_config.turnFirst && chassis.getDrivebaseType()!=DrivebaseType::XDRIVE) {
        double start_theta = l_config.backwards ? trajectory[0].heading + M_PI : trajectory[0].heading;
        double gps_target = lemlib::radToDeg(M_PI_2 - start_theta);
        chassis.turnToHeading(gps_target, 1000);
        chassis.waitUntilDone();
    }

    if(cancel_request) {result=MotionResult::Cancelled;return;}
    if(chassis.getDrivebaseType()==DrivebaseType::XDRIVE){
        auto config=l_config.xdriveControl;
        const auto hardware=nationals::configuredCascade();
        config.radius=hardware.radius;config.maxWheelSpeed=hardware.maxWheelSpeed;
        config.kS=velocityConfig.KS_straight;config.kA=velocityConfig.KA_straight;
        config.kV=velocityConfig.kV;config.wheelKp=velocityConfig.KP_straight;config.wheelKi=velocityConfig.KI_straight;
        config.maxVoltage=velocityConfig.max_voltage;
        if(l_config.turnFirst && !l_config.test){
            const auto p=chassis.getPose(true);
            const nationals::Pose start{p.x*INCH_TO_METER,p.y*INCH_TO_METER,p.theta};
            const nationals::Pose end{start.x,start.y,M_PI_2-trajectory.front().heading+(l_config.backwards?M_PI:0)};
            const double seconds=nationals::duration(start,end);
            result=nationals::followReference(leftMotors,rightMotors,velocityConfig.wheelDiameterMeters,
                velocityConfig.externalGearRatio,config,seconds,seconds+l_config.settleTimeout,
                [&](double time){return nationals::pointReference(start,end,time,seconds);},
                [&]{return cancel_request.load();});
            if(result!=MotionResult::Settled)return;
            result=MotionResult::Running;
        }
        size_t index=0; auto previousPose=chassis.getPose();
        result=nationals::followReference(leftMotors,rightMotors,velocityConfig.wheelDiameterMeters,
            velocityConfig.externalGearRatio,config,trajectory.back().time,
            trajectory.back().time+l_config.settleTimeout,[&](double time){
                while(index+1<trajectory.size()&&trajectory[index+1].time<=time)++index;
                auto s=trajectory[index];
                if(index+1<trajectory.size()){
                    const auto& b=trajectory[index+1];double q=std::clamp((time-s.time)/(b.time-s.time),0.0,1.0);
                    s.x+=(b.x-s.x)*q;s.y+=(b.y-s.y)*q;s.heading+=std::remainder(b.heading-s.heading,2*M_PI)*q;
                    s.linear_vel+=(b.linear_vel-s.linear_vel)*q;s.angular_vel+=(b.angular_vel-s.angular_vel)*q;
                }
                if(time>=trajectory.back().time){s.linear_vel=0;s.angular_vel=0;}
                const double h=M_PI_2-s.heading+(l_config.backwards?M_PI:0);
                auto p=chassis.getPose();distance_traveled_inches+=previousPose.distance(p);previousPose=p;
                return nationals::Reference{{s.x,s.y,h},s.linear_vel*std::cos(s.heading),s.linear_vel*std::sin(s.heading),-s.angular_vel};
            },[&]{return cancel_request.load();});
        return;
    }
    driveOutput().configure(&leftMotors,&rightMotors);
    const uint32_t token=driveOutput().acquire();
    if(!token) {result=getOdomStatus().valid?MotionResult::Busy:MotionResult::SensorFault;return;}
    struct Release {uint32_t token;~Release(){driveOutput().release(token);}} release{token};
    std::vector<std::string> logs;
    int trajectory_size = trajectory.size();
    uint32_t prev_time = pros::millis();
    
    double sum_lat_error = 0;
    double sum_head_error = 0;
    double sum_oscillation = 0; 
    float prev_w_cmd = 0;
    int steps = 0;

    lemlib::Pose start_pose = chassis.getPose();

    Eigen::MatrixXf cached_K(2, 3);
    cached_K.setZero();
    float last_solve_v = -9999.0f;
    float last_solve_w = -9999.0f;
    float last_solve_dt = -1;

    controller.reset();

    uint32_t loop_time = pros::millis();

    const uint32_t startTime=pros::millis();
    bool settling=false;uint32_t settleStart=0;
    result=MotionResult::TimedOut;
    for(int i=0;;) {
        if(cancel_request || !driveOutput().owns(token)) {result=MotionResult::Cancelled;break;}
        if(!getOdomStatus().valid) {result=MotionResult::SensorFault;break;}
        uint32_t current_time=pros::millis();
        const double elapsed=(current_time-startTime)*0.001;
        if(elapsed>trajectory.back().time+l_config.settleTimeout) break;
        double measured_dt=(current_time-prev_time)*0.001;
        if(measured_dt<=0) measured_dt=0.01;
        if(measured_dt>0.1) {result=MotionResult::SensorFault;break;}
        prev_time=current_time;
        while(i+1<trajectory_size && trajectory[i+1].time<=elapsed) ++i;
        State target_state=trajectory[i];
        if(i+1<trajectory_size) {
            const auto& b=trajectory[i+1];const double q=std::clamp((elapsed-target_state.time)/(b.time-target_state.time),0.0,1.0);
            target_state.x+=(b.x-target_state.x)*q;target_state.y+=(b.y-target_state.y)*q;
            target_state.heading+=std::remainder(b.heading-target_state.heading,2*M_PI)*q;
            target_state.linear_vel+=(b.linear_vel-target_state.linear_vel)*q;
            target_state.angular_vel+=(b.angular_vel-target_state.angular_vel)*q;
        }
        lemlib::Pose current_pose=chassis.getPose(true);
        distance_traveled_inches+=start_pose.distance(current_pose);start_pose=current_pose;
        const auto velocity=getLocalSpeed(true);
        if(elapsed>=trajectory.back().time) {
            target_state.linear_vel=0;target_state.angular_vel=0;
            const float heading=M_PI_2-current_pose.theta+(l_config.backwards?M_PI:0);
            const bool close=std::hypot(target_state.x-current_pose.x*INCH_TO_METER,target_state.y-current_pose.y*INCH_TO_METER)<l_config.positionTolerance &&
                std::abs(std::remainder(target_state.heading-heading,2*M_PI))<l_config.headingTolerance &&
                std::hypot(velocity.x,velocity.y)<1.0 && std::abs(velocity.theta)<0.1;
            if(close) {if(!settling){settling=true;settleStart=current_time;}if(current_time-settleStart>=150){result=MotionResult::Settled;break;}}
            else settling=false;
        }
        current_pose.x *= INCH_TO_METER;
        current_pose.y *= INCH_TO_METER;
        
        double velocity_scale = 1.0;
        double q_gain_mult = 1.0;
        double r_vel_mult = 1.0;
        double q_x_boost = 1.0;
        
        float q_x_effective = (l_config.backwards) ? l_config.q_x_b : l_config.q_x;
        float q_y_effective = (l_config.backwards) ? l_config.q_y_b : l_config.q_y;
        float q_theta_effective = (l_config.backwards) ? l_config.q_theta_b : l_config.q_theta;
        float r_ang_effective = (l_config.backwards) ? l_config.r_ang_b : l_config.r_ang;
        float r_vel_effective = (l_config.backwards) ? l_config.r_vel_b : l_config.r_vel;

        Eigen::Matrix3f Q_mat; 
        Q_mat << q_x_effective * q_gain_mult * q_x_boost * l_config.q_scalar, 0, 0,
                 0, q_y_effective * q_gain_mult * l_config.q_scalar, 0,
                 0, 0, q_theta_effective * q_gain_mult * l_config.q_scalar;
        
        Eigen::Matrix2f R_mat;
        R_mat << r_vel_effective * r_vel_mult, 0,
                 0, r_ang_effective;

        double math_theta = M_PI_2 - current_pose.theta; 
        double effective_theta = l_config.backwards ? math_theta + M_PI : math_theta;
        double target_heading = target_state.heading;
        double errorTheta = angleError(effective_theta, target_heading);
        
        Eigen::Vector3d global_error;
        global_error << target_state.x - current_pose.x, target_state.y - current_pose.y, errorTheta;
        
        Eigen::Matrix3d rotation_matrix;
        rotation_matrix <<  std::cos(effective_theta), std::sin(effective_theta), 0, 
                           -std::sin(effective_theta), std::cos(effective_theta), 0, 
                            0, 0, 1;
        Eigen::Vector3d error = rotation_matrix * global_error;
        float v_ref = std::abs(target_state.linear_vel) * velocity_scale;
        float w_ref = target_state.angular_vel * velocity_scale;
        float a_v_ref = (v_ref < 0.15f) ? 0.15f : v_ref;
        constexpr float eps = -1e-3f;
        
        // Fast DARE cache check: if (v_ref, w_ref) close to last solve, reuse K to save CPU
        if (std::abs(v_ref - last_solve_v) > 0.05f || std::abs(w_ref - last_solve_w) > 0.05f || std::abs(measured_dt-last_solve_dt)>0.001f) {
            Eigen::Matrix3f A;
            A << eps, w_ref, 0,
                -w_ref, eps, a_v_ref, 
                 0, 0, eps;
                 
            Eigen::Matrix<float, 3, 2> B;
            B << -1, 0,
                  0, 0,
                  0, -1;
                
            auto discAB = discretizeAB(A, B, measured_dt);
            Eigen::MatrixXf X = dareSolver(discAB.first, discAB.second, Q_mat, R_mat);
            
            Eigen::Matrix2f R_reg = R_mat + Eigen::Matrix2f::Identity() * 1e-4f;
            cached_K = (R_reg + discAB.second.transpose() * X * discAB.second).ldlt().solve(discAB.second.transpose() * X * discAB.first);
            
            if (cached_K.hasNaN() || !cached_K.allFinite()) {
                result=MotionResult::InvalidInput; break;
            }

            last_solve_v = v_ref;
            last_solve_w = w_ref;
            last_solve_dt=measured_dt;
        }
        
        // Optimal control law: u = - K * e
        Eigen::Vector2f u = -cached_K * error.cast<float>();
        
        if (u.hasNaN() || !u.allFinite()) {
            result=MotionResult::InvalidInput; break;
        }

        float u_v = clamp(u(0), -l_config.max_lin_correction, l_config.max_lin_correction);
        float u_w = clamp(u(1), -l_config.max_ang_correction, l_config.max_ang_correction);
        
        float v_cmd = v_ref + u_v;
        float w_cmd = w_ref + u_w;

        if (l_config.backwards) {
            v_cmd = -v_cmd;
        }

        double lat_err = std::abs(error(1));
        double head_err = std::abs(error(2));
        double instant_oscillation = std::abs(w_cmd - prev_w_cmd);
        prev_w_cmd = w_cmd;

        sum_lat_error += lat_err;
        sum_head_error += head_err;
        sum_oscillation += instant_oscillation;
        steps++;

        // Read motor velocities for closed-loop velocity tracking
        float left_actual_mps = leftMotors.get_actual_velocity(0) * rpm_to_mps_factor;
        float right_actual_mps = rightMotors.get_actual_velocity(0) * rpm_to_mps_factor;

        DrivetrainVoltages output_voltages = controller.update(
            v_cmd, w_cmd, left_actual_mps, right_actual_mps, measured_dt
        );

        output_voltages.rightVoltage = clamp(output_voltages.rightVoltage, -12.0, 12.0);
        output_voltages.leftVoltage = clamp(output_voltages.leftVoltage, -12.0, 12.0);

        if(!std::isfinite(left_actual_mps)||!std::isfinite(right_actual_mps)) {result=MotionResult::SensorFault;break;}
        if(!driveOutput().voltage(token,output_voltages.leftVoltage,output_voltages.rightVoltage)) {result=MotionResult::SensorFault;break;}
        
        if (l_config.log && i % 5 == 0) {
            std::ostringstream ss;
            ss << Vector2(current_pose.x, current_pose.y).latex() << ",";
            if(logs.size()<512) logs.push_back(ss.str());
        }
        
        pros::Task::delay_until(&loop_time, 10);
    }

    driveOutput().release(token);

    if (steps == 0) steps = 1; 
    double avg_lat_error = sum_lat_error / steps;
    double avg_head_error = sum_head_error / steps;
    double avg_jerk = sum_oscillation / steps;

    if (l_config.log) {
        std::cout << "\n--- LTV PERFORMANCE SUMMARY ---" << std::endl;
        std::cout << "Steps Completed: " << steps << " / " << trajectory_size << std::endl;
        std::cout << "Avg Lateral Error: " << (avg_lat_error / INCH_TO_METER) << " in" << std::endl;
        std::cout << "Avg Heading Error: " << lemlib::radToDeg(avg_head_error) << " deg" << std::endl;
        std::cout << "Avg Control Jerk:  " << avg_jerk << std::endl;
    }

    is_running = false;
}

Eigen::MatrixXf LTVPathFollower::dareSolver(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B, const Eigen::MatrixXf &Q, const Eigen::MatrixXf &R) {
    int states = A.rows();
    
    Eigen::MatrixXf A_k = A;
    Eigen::MatrixXf R_reg = R + Eigen::MatrixXf::Identity(R.rows(), R.cols()) * 1e-4f;
    Eigen::MatrixXf G_k = B * R_reg.ldlt().solve(B.transpose()); 
    Eigen::MatrixXf H_k;
    Eigen::MatrixXf H_k1 = Q;
    
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(states, states);

    for (int i = 0; i < 40; ++i) {
        H_k = H_k1;
        Eigen::MatrixXf W = I + G_k * H_k;
        auto W_solver = W.partialPivLu();
        Eigen::MatrixXf V_1 = W_solver.solve(A_k);
        Eigen::MatrixXf V_2 = W_solver.solve(G_k);

        G_k += A_k * V_2 * A_k.transpose();
        H_k1 = H_k + V_1.transpose() * H_k * A_k;
        A_k *= V_1;
        
        if (H_k1.hasNaN() || !H_k1.allFinite()) {
            return Q;
        }

        if ((H_k1 - H_k).norm() <= 1e-5f * H_k1.norm()) {
            break;
        }
    }

    return H_k1.allFinite() ? H_k1 : Q;
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> LTVPathFollower::discretizeAB(
    const Eigen::MatrixXf& contA, const Eigen::MatrixXf& contB, double dtSeconds) {
    if (dtSeconds <= 0.0001) dtSeconds = 0.01;
    int states = contA.rows();
    int inputs = contB.cols();
    Eigen::MatrixXf M(states + inputs, states + inputs);
    M.setZero();
    M.topLeftCorner(states, states) = contA;
    M.topRightCorner(states, inputs) = contB;
    Eigen::MatrixXf Mdt = M * dtSeconds;
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(M.rows(), M.cols());
    Eigen::MatrixXf M2 = Mdt * Mdt;
    Eigen::MatrixXf phi = I + Mdt + (M2 * 0.5f); 
    Eigen::MatrixXf discA = phi.topLeftCorner(states, states);
    Eigen::MatrixXf discB = phi.topRightCorner(states, inputs);
    return {discA, discB};
}

void LTVPathFollower::precompute_paths(const std::vector<std::string>& path_names) {
    Lock guard(cacheMutex);
    for(const auto& data:path_names) precomputed_paths[data]=prepare_trajectory(data);
}
void LTVPathFollower::precompute_paths_task(void*) {}

std::vector<std::vector<double>> LTVPathFollower::parse_tuples(const std::string& line) {
    std::vector<std::vector<double>> result;
    std::string temp;
    bool inside_parens = false;
    
    for (char c : line) {
        if (c == '(') {
            temp.clear();
            inside_parens = true;
        } else if (c == ')') { 
            std::replace(temp.begin(), temp.end(), ',', ' ');
            std::istringstream ss(temp);
            std::vector<double> tuple;
            double val;
            
            while (ss >> val) {
                tuple.push_back(val);
            }
            
            result.push_back(tuple);
            inside_parens = false;
        } else if (inside_parens) {
            temp += c;
        }
    }
    return result;
}

std::vector<State> LTVPathFollower::prepare_trajectory(const std::string& data) {
    if(data.size()>1024*1024) return {};
    std::istringstream ss(data);
    std::vector<std::vector<double>> P, V;
    std::string line;
    
    while (std::getline(ss, line)) {
        if (line.find("P =") != std::string::npos) {
            if(line.find('{')==std::string::npos) return {};
            P = parse_tuples(line.substr(line.find('{')));
        } else if (line.find("V =") != std::string::npos) {
            if(line.find('{')==std::string::npos) return {};
            V = parse_tuples(line.substr(line.find('{')));
        }
    }
    
    if(P.size()!=V.size()) return {};
    size_t n = P.size();
    if (n == 0) return {};
    
    std::vector<State> states(n);
    for (size_t i = 0; i < n; i++) {
        if(P[i].size()!=3||V[i].size()!=2) return {};
        for(double v:P[i]) if(!std::isfinite(v)) return {};
        for(double v:V[i]) if(!std::isfinite(v)) return {};
        states[i].time=i*0.01; // Legacy text format explicitly uses 100Hz.
        if (P[i].size() >= 3) {
            states[i].x = P[i][0];
            states[i].y = P[i][1];
            states[i].heading = P[i][2];
        }
        
        if (V[i].size() >= 2) {
            states[i].linear_vel = V[i][0];
            states[i].angular_vel = V[i][1];
        }
    }
    
    return states;
}

} // namespace lemlib
