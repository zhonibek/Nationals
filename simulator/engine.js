/**
 * ============================================================================
 * IRAlib VEX V5RC Override Robot & Field Emulator — Core Engine
 * 2D/3D Viewport, Holonomic X-Drive, Jerry.io Visual Waypoint Planner,
 * LTV DARE Riccati Solver, Quintic Hermite Splines, EKF & Autonomous Engine
 * ============================================================================
 */

// Math & Conversion Constants
const INCH_TO_METER = 0.0254;
const METER_TO_INCH = 1.0 / 0.0254;
const DEG_TO_RAD = Math.PI / 180.0;
const RAD_TO_DEG = 180.0 / Math.PI;

function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

function normalizeAngle(rad) {
  while (rad > Math.PI) rad -= 2.0 * Math.PI;
  while (rad < -Math.PI) rad += 2.0 * Math.PI;
  return rad;
}

function sgn(val) {
  if (val > 0) return 1;
  if (val < 0) return -1;
  return 0;
}

// ============================================================================
// 1. Matrix & Linear Algebra (3x3, 2x2, 5x5) for DARE Solver & EKF
// ============================================================================
class Matrix {
  constructor(rows, cols, data = null) {
    this.rows = rows;
    this.cols = cols;
    this.data = new Float64Array(rows * cols);
    if (data) {
      for (let i = 0; i < Math.min(this.data.length, data.length); i++) {
        this.data[i] = data[i];
      }
    }
  }

  static identity(n) {
    const m = new Matrix(n, n);
    for (let i = 0; i < n; i++) m.set(i, i, 1.0);
    return m;
  }

  static zeros(rows, cols) {
    return new Matrix(rows, cols);
  }

  static diag(arr) {
    const m = new Matrix(arr.length, arr.length);
    for (let i = 0; i < arr.length; i++) m.set(i, i, arr[i]);
    return m;
  }

  get(r, c) { return this.data[r * this.cols + c]; }
  set(r, c, val) { this.data[r * this.cols + c] = val; }

  clone() {
    const m = new Matrix(this.rows, this.cols);
    m.data.set(this.data);
    return m;
  }

  transpose() {
    const res = new Matrix(this.cols, this.rows);
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        res.set(c, r, this.get(r, c));
      }
    }
    return res;
  }

  add(other) {
    const res = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) res.data[i] = this.data[i] + other.data[i];
    return res;
  }

  sub(other) {
    const res = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) res.data[i] = this.data[i] - other.data[i];
    return res;
  }

  mulScalar(s) {
    const res = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) res.data[i] = this.data[i] * s;
    return res;
  }

  multiply(other) {
    if (this.cols !== other.rows) throw new Error("Matrix dimension mismatch");
    const res = new Matrix(this.rows, other.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < other.cols; j++) {
        let sum = 0;
        for (let k = 0; k < this.cols; k++) {
          sum += this.get(i, k) * other.get(k, j);
        }
        res.set(i, j, sum);
      }
    }
    return res;
  }

  solve(B) {
    const n = this.rows;
    const m = B.cols;
    const A = this.clone();
    const X = B.clone();

    for (let i = 0; i < n; i++) {
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(A.get(k, i)) > Math.abs(A.get(maxRow, i))) maxRow = k;
      }
      for (let k = i; k < n; k++) {
        const tmp = A.get(i, k);
        A.set(i, k, A.get(maxRow, k));
        A.set(maxRow, k, tmp);
      }
      for (let k = 0; k < m; k++) {
        const tmp = X.get(i, k);
        X.set(i, k, X.get(maxRow, k));
        X.set(maxRow, k, tmp);
      }

      const pivot = A.get(i, i);
      if (Math.abs(pivot) < 1e-12) continue;

      for (let j = i + 1; j < n; j++) {
        const factor = A.get(j, i) / pivot;
        for (let k = i; k < n; k++) {
          A.set(j, k, A.get(j, k) - factor * A.get(i, k));
        }
        for (let k = 0; k < m; k++) {
          X.set(j, k, X.get(j, k) - factor * X.get(i, k));
        }
      }
    }

    for (let i = n - 1; i >= 0; i--) {
      const pivot = A.get(i, i);
      if (Math.abs(pivot) < 1e-12) continue;
      for (let k = 0; k < m; k++) {
        let sum = X.get(i, k);
        for (let j = i + 1; j < n; j++) {
          sum -= A.get(i, j) * X.get(j, k);
        }
        X.set(i, k, sum / pivot);
      }
    }
    return X;
  }

  inverse() {
    return this.solve(Matrix.identity(this.rows));
  }

  norm() {
    let sum = 0;
    for (let i = 0; i < this.data.length; i++) sum += this.data[i] * this.data[i];
    return Math.sqrt(sum);
  }
}

// ============================================================================
// 2. DARE Riccati Solver & Discretization (LTV-LQR Engine)
// ============================================================================
class LTVMath {
  static dareSolver(A, B, Q, R) {
    const states = A.rows;
    let A_k = A.clone();
    const R_reg = R.add(Matrix.identity(R.rows).mulScalar(1e-4));
    let G_k = B.multiply(R_reg.inverse()).multiply(B.transpose());
    let H_k = Matrix.zeros(states, states);
    let H_k1 = Q.clone();
    const I = Matrix.identity(states);

    for (let iter = 0; iter < 30; iter++) {
      H_k = H_k1.clone();
      const W = I.add(G_k.multiply(H_k));
      const V_1 = W.solve(A_k);
      const V_2 = W.solve(G_k);

      G_k = G_k.add(A_k.multiply(V_2).multiply(A_k.transpose()));
      H_k1 = H_k.add(V_1.transpose().multiply(H_k).multiply(A_k));
      A_k = A_k.multiply(V_1);

      if (H_k1.sub(H_k).norm() <= 1e-5 * H_k1.norm()) {
        break;
      }
    }
    return H_k1;
  }

  static discretizeAB(contA, contB, dt) {
    if (dt <= 0.0001) dt = 0.01;
    const states = contA.rows;
    const inputs = contB.cols;
    const total = states + inputs;
    const M = Matrix.zeros(total, total);

    for (let r = 0; r < states; r++) {
      for (let c = 0; c < states; c++) M.set(r, c, contA.get(r, c));
      for (let c = 0; c < inputs; c++) M.set(r, states + c, contB.get(r, c));
    }

    const Mdt = M.mulScalar(dt);
    const M2 = Mdt.multiply(Mdt).mulScalar(0.5);
    const phi = Matrix.identity(total).add(Mdt).add(M2);

    const discA = Matrix.zeros(states, states);
    const discB = Matrix.zeros(states, inputs);

    for (let r = 0; r < states; r++) {
      for (let c = 0; c < states; c++) discA.set(r, c, phi.get(r, c));
      for (let c = 0; c < inputs; c++) discB.set(r, c, phi.get(r, states + c));
    }

    return { discA, discB };
  }
}

// ============================================================================
// 3. Quintic Hermite Spline Generator (Matches QuinticSpline.cpp commit 65f6ec1)
// ============================================================================
class QuinticSplineGenerator {
  static generateTrajectory(startPose, endPose, maxVel = 1.0, maxAccel = 1.8, maxJerk = 3.5, dt = 0.01) {
    if (dt <= 1e-4) dt = 0.01;

    const x0 = startPose.x * INCH_TO_METER;
    const y0 = startPose.y * INCH_TO_METER;
    const theta0 = startPose.theta * DEG_TO_RAD;

    const x1 = endPose.x * INCH_TO_METER;
    const y1 = endPose.y * INCH_TO_METER;
    const theta1 = endPose.theta * DEG_TO_RAD;

    const dist = Math.hypot(x1 - x0, y1 - y0);
    if (dist < 1e-3) {
      return [{ x: x0, y: y0, heading: Math.PI / 2 - theta0, linear_vel: 0, angular_vel: 0 }];
    }

    const scale = Math.max(1.35 * dist, 0.45);
    const vx0 = scale * Math.sin(theta0);
    const vy0 = scale * Math.cos(theta0);
    const vx1 = scale * Math.sin(theta1);
    const vy1 = scale * Math.cos(theta1);

    const ax0 = 0.0, ay0 = 0.0;
    const ax1 = 0.0, ay1 = 0.0;

    const cx0 = x0;
    const cx1 = vx0;
    const cx2 = 0.5 * ax0;
    const cx3 = 10.0 * (x1 - x0) - (6.0 * vx0 + 4.0 * vx1) - (1.5 * ax0 - 0.5 * ax1);
    const cx4 = -15.0 * (x1 - x0) + (8.0 * vx0 + 7.0 * vx1) + (1.5 * ax0 - ax1);
    const cx5 = 6.0 * (x1 - x0) - 3.0 * (vx0 + vx1) - 0.5 * (ax0 - ax1);

    const cy0 = y0;
    const cy1 = vy0;
    const cy2 = 0.5 * ay0;
    const cy3 = 10.0 * (y1 - y0) - (6.0 * vy0 + 4.0 * vy1) - (1.5 * ay0 - 0.5 * ay1);
    const cy4 = -15.0 * (y1 - y0) + (8.0 * vy0 + 7.0 * vy1) + (1.5 * ay0 - ax1);
    const cy5 = 6.0 * (y1 - y0) - 3.0 * (vy0 + vy1) - 0.5 * (ax0 - ax1);

    const effectiveMaxVel = Math.max(maxVel, 0.2);
    const totalTime = (1.5 * dist / effectiveMaxVel) + 0.6;
    const numSteps = Math.max(Math.ceil(totalTime / dt), 10);

    const trajectory = [];
    let prevHeading = Math.PI / 2 - theta0;

    for (let i = 0; i <= numSteps; i++) {
      const t = i / numSteps;
      const t2 = t * t;
      const t3 = t2 * t;
      const t4 = t3 * t;
      const t5 = t4 * t;

      const px = cx0 + cx1 * t + cx2 * t2 + cx3 * t3 + cx4 * t4 + cx5 * t5;
      const py = cy0 + cy1 * t + cy2 * t2 + cy3 * t3 + cy4 * t4 + cy5 * t5;

      const dpx = cx1 + 2.0 * cx2 * t + 3.0 * cx3 * t2 + 4.0 * cx4 * t3 + 5.0 * cx5 * t4;
      const dpy = cy1 + 2.0 * cy2 * t + 3.0 * cy3 * t2 + 4.0 * cy4 * t3 + 5.0 * cy5 * t4;

      let heading = Math.atan2(dpy, dpx);
      if (isNaN(heading)) heading = prevHeading;

      const s_vel = Math.sin(Math.PI * t);
      const v = maxVel * s_vel;

      let w = 0.0;
      if (i > 0) {
        let dTheta = heading - prevHeading;
        while (dTheta > Math.PI) dTheta -= 2.0 * Math.PI;
        while (dTheta < -Math.PI) dTheta += 2.0 * Math.PI;
        w = dTheta / dt;
      }
      prevHeading = heading;

      trajectory.push({ x: px, y: py, heading: heading, linear_vel: v, angular_vel: w });
    }
    return trajectory;
  }

  static generateMultiPointTrajectory(waypoints, maxVel = 1.0, maxAccel = 1.8, dt = 0.01) {
    if (waypoints.length < 2) return [];
    const full = [];
    for (let i = 0; i < waypoints.length - 1; i++) {
      const seg = QuinticSplineGenerator.generateTrajectory(waypoints[i], waypoints[i + 1], maxVel, maxAccel, 3.5, dt);
      if (i > 0 && seg.length > 0) seg.shift();
      full.push(...seg);
    }
    return full;
  }
}

// ============================================================================
// 4. Extended Kalman Filter (5-State RobotEKF matching EKF.cpp)
// ============================================================================
class RobotEKF {
  constructor(initialPose = { x: 0, y: 0, theta: 0 }) {
    this.reset(initialPose);
  }

  reset(pose) {
    this.x = pose.x * INCH_TO_METER;
    this.y = pose.y * INCH_TO_METER;
    this.theta = pose.theta * DEG_TO_RAD;
    this.v = 0.0;
    this.w = 0.0;
    this.P = Matrix.diag([0.01, 0.01, 0.005, 0.1, 0.1]);
    this.Q = Matrix.diag([0.0001, 0.0001, 0.00005, 0.01, 0.01]);
  }

  predict(dt) {
    if (dt <= 1e-4) dt = 0.01;
    this.x += this.v * Math.sin(this.theta) * dt;
    this.y += this.v * Math.cos(this.theta) * dt;
    this.theta = normalizeAngle(this.theta + this.w * dt);

    const F = Matrix.identity(5);
    F.set(0, 2, this.v * Math.cos(this.theta) * dt);
    F.set(0, 3, Math.sin(this.theta) * dt);
    F.set(1, 2, -this.v * Math.sin(this.theta) * dt);
    F.set(1, 3, Math.cos(this.theta) * dt);
    F.set(2, 4, dt);

    this.P = F.multiply(this.P).multiply(F.transpose()).add(this.Q);
  }

  updatePose(odomXMeters, odomYMeters, odomThetaRad, stdDevPos = 0.03, stdDevTheta = 0.02) {
    const H = Matrix.zeros(3, 5);
    H.set(0, 0, 1.0);
    H.set(1, 1, 1.0);
    H.set(2, 2, 1.0);

    const R = Matrix.diag([stdDevPos * stdDevPos, stdDevPos * stdDevPos, stdDevTheta * stdDevTheta]);
    let dyTheta = normalizeAngle(odomThetaRad - this.theta);
    const y = new Matrix(3, 1, [odomXMeters - this.x, odomYMeters - this.y, dyTheta]);

    const S = H.multiply(this.P).multiply(H.transpose()).add(R);
    const K = this.P.multiply(H.transpose()).multiply(S.inverse());

    const update = K.multiply(y);
    this.x += update.get(0, 0);
    this.y += update.get(1, 0);
    this.theta = normalizeAngle(this.theta + update.get(2, 0));
    this.v += update.get(3, 0);
    this.w += update.get(4, 0);

    const I = Matrix.identity(5);
    this.P = I.sub(K.multiply(H)).multiply(this.P);
  }

  getPose() {
    return {
      x: this.x * METER_TO_INCH,
      y: this.y * METER_TO_INCH,
      theta: this.theta * RAD_TO_DEG
    };
  }
}

// ============================================================================
// 5. Controllers: VelocityController, LQR & PID
// ============================================================================
class VelocityController {
  constructor(config) {
    this.config = config;
    this.reset();
  }

  reset() {
    this.prev_v_cmd = 0.0;
    this.prev_w_cmd = 0.0;
    this.left_integral = 0.0;
    this.right_integral = 0.0;
  }

  update(v_cmd, w_cmd, left_actual_mps, right_actual_mps, dt = 0.01) {
    const halfTrack = this.config.trackWidthMeters / 2.0;
    const v_left_target = v_cmd - (w_cmd * halfTrack);
    const v_right_target = v_cmd + (w_cmd * halfTrack);

    const a_lin = (v_cmd - this.prev_v_cmd) / dt;
    const a_ang = (w_cmd - this.prev_w_cmd) / dt;
    this.prev_v_cmd = v_cmd;
    this.prev_w_cmd = w_cmd;

    const a_left_target = a_lin - (a_ang * halfTrack);
    const a_right_target = a_lin + (a_ang * halfTrack);

    const turnRatio = (Math.abs(v_cmd) + Math.abs(w_cmd) > 1e-4) ?
      Math.abs(w_cmd) / (Math.abs(v_cmd) + Math.abs(w_cmd)) : 0.0;
    const kS_eff = this.config.KS_straight * (1.0 - turnRatio) + this.config.KS_turn * turnRatio;
    const kA_eff = this.config.KA_straight * (1.0 - turnRatio) + this.config.KA_turn * turnRatio;

    const ff_left = (kS_eff * Math.tanh(v_left_target / 0.05)) +
                    (this.config.kV * v_left_target) +
                    (kA_eff * a_left_target);

    const ff_right = (kS_eff * Math.tanh(v_right_target / 0.05)) +
                     (this.config.kV * v_right_target) +
                     (kA_eff * a_right_target);

    const err_left = v_left_target - left_actual_mps;
    const err_right = v_right_target - right_actual_mps;

    this.left_integral = clamp(this.left_integral + err_left * dt, -2.0, 2.0);
    this.right_integral = clamp(this.right_integral + err_right * dt, -2.0, 2.0);

    const fb_left = (this.config.KP_straight * err_left) + (this.config.KI_straight * this.left_integral);
    const fb_right = (this.config.KP_straight * err_right) + (this.config.KI_straight * this.right_integral);

    let total_left = ff_left + fb_left;
    let total_right = ff_right + fb_right;

    let slipTriggered = false;
    if (this.config.enableTCS) {
      if (Math.abs(v_left_target) > 0.3) {
        const slipLeft = Math.abs(err_left) / Math.abs(v_left_target);
        if (slipLeft > this.config.maxSlipRatio && (total_left * v_left_target > 0)) {
          total_left *= clamp(1.0 - (slipLeft - this.config.maxSlipRatio) * 1.5, 0.4, 1.0);
          slipTriggered = true;
        }
      }
      if (Math.abs(v_right_target) > 0.3) {
        const slipRight = Math.abs(err_right) / Math.abs(v_right_target);
        if (slipRight > this.config.maxSlipRatio && (total_right * v_right_target > 0)) {
          total_right *= clamp(1.0 - (slipRight - this.config.maxSlipRatio) * 1.5, 0.4, 1.0);
          slipTriggered = true;
        }
      }
    }

    total_left = clamp(total_left, -this.config.max_voltage, this.config.max_voltage);
    total_right = clamp(total_right, -this.config.max_voltage, this.config.max_voltage);

    return { leftVoltage: total_left, rightVoltage: total_right, slipTriggered };
  }
}

class LemLibLQRController {
  constructor(settings) {
    this.settings = settings;
    this.reset();
  }

  reset() {
    this.integral = 0.0;
    this.prevError = 0.0;
    this.filteredVelocity = 0.0;
    this.isFirstStep = true;
    this.timeSpentSmallError = 0.0;
    this.timeSpentLargeError = 0.0;
    this.output = 0.0;
    this.isSettled = false;
  }

  update(error, dt = 0.01) {
    if (dt <= 0) dt = 0.01;
    if (this.isFirstStep) {
      this.prevError = error;
      this.filteredVelocity = 0.0;
      this.isFirstStep = false;
    } else {
      const derivative = (this.prevError - error) / dt;
      this.filteredVelocity = (0.75 * derivative) + (0.25 * this.filteredVelocity);
      this.prevError = error;
    }

    const velocity = this.filteredVelocity;
    const accel = 0.0;

    if (this.settings.windupRange === 0 || Math.abs(error) <= this.settings.windupRange) {
      this.integral = clamp(this.integral + error * dt, -30.0, 30.0);
    } else {
      this.integral = 0.0;
    }

    const predictedVelocity = velocity + accel * dt;
    const predictedError = error - (velocity * dt) - (0.5 * accel * dt * dt);

    let rawOut = (this.settings.kP * predictedError) -
                 (this.settings.kV * predictedVelocity) -
                 (this.settings.kA * accel) +
                 (this.settings.kI * this.integral);

    if (this.settings.slew && this.settings.slew > 0) {
      const maxDelta = this.settings.slew * dt * 12.0;
      this.output += clamp(rawOut - this.output, -maxDelta, maxDelta);
    } else {
      this.output = rawOut;
    }

    // Exit conditions matching LemLib settings in main.cpp
    const absErr = Math.abs(error);
    if (absErr < (this.settings.smallError || 0.8)) {
      this.timeSpentSmallError += dt * 1000;
    } else {
      this.timeSpentSmallError = 0;
    }

    if (absErr < (this.settings.largeError || 2.5)) {
      this.timeSpentLargeError += dt * 1000;
    } else {
      this.timeSpentLargeError = 0;
    }

    const smallTimeout = this.settings.smallErrorTimeout || 100;
    const largeTimeout = this.settings.largeErrorTimeout || 450;

    const isStationary = Math.abs(this.filteredVelocity || 0) < 0.06;
    if (this.timeSpentSmallError >= smallTimeout || (this.timeSpentLargeError >= largeTimeout && isStationary)) {
      this.isSettled = true;
    }

    return clamp(this.output, -12.0, 12.0);
  }
}

class LemLibPIDController {
  constructor(settings) {
    this.settings = settings;
    this.reset();
  }

  reset() {
    this.prevError = 0.0;
    this.integral = 0.0;
    this.timeSpentSmallError = 0.0;
    this.timeSpentLargeError = 0.0;
    this.output = 0.0;
    this.isSettled = false;
    this.isFirstStep = true;
  }

  update(error, dt = 0.01) {
    if (dt <= 0) dt = 0.01;

    const absErr = Math.abs(error);

    // Deadband check
    if (this.settings.deadband && absErr < this.settings.deadband) {
      this.output = 0.0;
      this.prevError = error;
      return 0.0;
    }

    // Integral with windup range and sign-flip reset
    if (this.settings.signFlipReset && (error * this.prevError < 0)) {
      this.integral = 0.0;
    }

    if (!this.settings.windupRange || absErr <= this.settings.windupRange) {
      this.integral = clamp(this.integral + error * dt, -25.0, 25.0);
    } else {
      this.integral = 0.0;
    }

    let derivative = 0.0;
    if (this.isFirstStep) {
      this.prevError = error;
      this.isFirstStep = false;
    } else {
      derivative = (error - this.prevError) / dt;
      this.prevError = error;
    }

    let rawOut = (this.settings.kP * error) + (this.settings.kI * this.integral) + (this.settings.kD * derivative);

    // Slew rate limiting in Volts/sec (e.g. 25.0 V/s)
    const slew = this.settings.slew || this.settings.slewRate;
    if (slew && slew > 0) {
      const maxDelta = slew * dt;
      this.output += clamp(rawOut - this.output, -maxDelta, maxDelta);
    } else {
      this.output = rawOut;
    }

    if (absErr < (this.settings.smallError || 0.8)) {
      this.timeSpentSmallError += dt * 1000;
    } else {
      this.timeSpentSmallError = 0;
    }

    if (absErr < (this.settings.largeError || 2.5)) {
      this.timeSpentLargeError += dt * 1000;
    } else {
      this.timeSpentLargeError = 0;
    }

    const smallTimeout = this.settings.smallErrorTimeout || 100;
    const largeTimeout = this.settings.largeErrorTimeout || 450;

    const isStationary = Math.abs(derivative) < 0.08;
    if (this.timeSpentSmallError >= smallTimeout || (this.timeSpentLargeError >= largeTimeout && isStationary)) {
      this.isSettled = true;
    }

    return clamp(this.output, -12.0, 12.0);
  }
}

const LQRController = LemLibLQRController;
const PIDController = LemLibPIDController;

// ============================================================================
// 6. Complete VEX Holonomic Robot Simulator Physics & Execution
// ===========================================================================// ============================================================================
// 6. Complete VEX Holonomic Robot Simulator Physics & Execution (Digital Twin)
// ============================================================================
class VexRobotSimulator {
  constructor(control = null, options = {}) {
    this.activeRobotId="red-1";this.matchMode=false;
    this.productionControl=control; this.lastMotionResult="Idle";
    this.commandedWheelVoltages=null; this.odomVelocity=[0,0,0];
    // True Physical Parameters (Rigid Body Dynamics)
    this.massKg = 6.8; // Base robot mass (~15 lbs with mechanisms)
    this.moiKgM2 = 0.145; // Moment of Inertia around vertical yaw axis (kg*m^2)
    this.trackWidthInches = 10.5;
    this.trackWidthM = this.trackWidthInches * INCH_TO_METER; // 0.2667 m
    this.rEff = this.trackWidthM / Math.SQRT2; // ~0.1886 m effective turning arm for 45 deg wheels
    this.wheelDiameterInches = 3.25; // Authentic competition 3.25" omni wheels
    this.wheelRadiusM = (this.wheelDiameterInches * INCH_TO_METER) / 2.0; // 0.041275 m
    this.wheelInertia = 0.0006; // kg*m^2 per wheel rotor

    const config=options.config ?? globalThis.ROBOT_CONFIG ?? VexRobotSimulator.dependencies?.config;
    this.config=config;
    this.Fleet=options.Fleet ?? globalThis.OverrideFleet ?? VexRobotSimulator.dependencies?.Fleet;
    if(config){
      this.trackWidthInches=config.widthIn;
      this.trackWidthM=config.widthIn*INCH_TO_METER;
      this.rEff=(config.widthIn+config.wheelbaseIn)*INCH_TO_METER/(2*Math.SQRT2);
      this.wheelDiameterInches=config.wheelDiameterIn;
      this.wheelRadiusM=config.wheelDiameterIn*INCH_TO_METER/2;
    }
    const rpm=config?.cartridgeRpm ?? 200, ratio=config?.externalGearRatio ?? 1;
    this.nominalWheelSpeed=rpm*ratio*2*Math.PI*this.wheelRadiusM/60;
    this.productionControl?.configure(this.rEff,this.nominalWheelSpeed);
    // Tire-Ground Contact & Traction Limits (Rubber on VEX foam tile)
    this.muLong = 0.85; // Static/kinetic friction limit on foam
    this.muLat = 0.05;  // Free-spinning roller lateral resistance
    this.kSlip = 650.0; // Tire tangential shear stiffness (N / (m/s))

    // V5 11W Smart Motor Electromechanics (4x 200 RPM green cartridges)
    this.motorKt = 0.10;          // Torque constant N*m / A
    this.motorKe = 0.573;         // Back-EMF constant V / (rad/s)
    this.motorR = 1.05;           // Armature resistance (Ohms)
    this.motorCurrentLimit = 3.5; // Max A per motor before breaker trip
    this.maxVoltageSlewPerSec = 48.0; // V/s voltage slew limit
    this.batteryVoltage = 12.6;   // Nominal full V5 battery
    this.batteryInternalR = 0.04; // Battery internal resistance (Ohms)

    // Physical State in Field Reference Frame (Inertial, 100% conserved)
    this.x = 0.0;
    this.y = 0.0;
    this.theta = 0.0; // 0 = North (+Y)
    this.Vx = 0.0;    // Field X velocity (m/s)
    this.Vy = 0.0;    // Field Y velocity (m/s)
    this.vx = 0.0;    // Body lateral speed (m/s)
    this.vy = 0.0;    // Body longitudinal speed (m/s)
    this.v = 0.0;     // Total speed (m/s)
    this.w = 0.0;     // Yaw rate (rad/s, CW positive)
    this.Ax = 0.0;    // Field X acceleration (m/s^2)
    this.Ay = 0.0;    // Field Y acceleration (m/s^2)
    this.ax = 0.0;    // Body lateral acceleration (m/s^2)
    this.ay = 0.0;    // Body forward acceleration (m/s^2)
    this.alpha = 0.0; // Angular acceleration (rad/s^2)

    // 4 Motors / Wheels: 0=FL, 1=BL, 2=FR, 3=BR
    this.wheelOmega = [0.0, 0.0, 0.0, 0.0]; // rad/s
    this.motorVolts = [0.0, 0.0, 0.0, 0.0]; // V
    this.motorCurrents = [0.0, 0.0, 0.0, 0.0]; // A
    this.motorTorques = [0.0, 0.0, 0.0, 0.0]; // N*m
    this.wheelTractionForces = [0.0, 0.0, 0.0, 0.0]; // N
    this.wheelSlips = [0.0, 0.0, 0.0, 0.0]; // m/s

    // Odometry & Sensor Suite (with realistic noise and drift)
    this.odom = { x: 0.0, y: 0.0, theta: 0.0 };
    this.ekf = new RobotEKF({ x: 0, y: 0, theta: 0 });
    this.ekfPose = { x: 0.0, y: 0.0, theta: 0.0 };
    this.gyroDriftRate = 0.015; // deg/s drift rate
    this.gyroBias = 0.0;

    // Mobile Goal Clamping Mechanics
    this.clampedGoalIndex = -1;
    this.isPneumaticClamped = false;

    // Motion feedback is exclusively the production C++ cascade.
    this.isRunning = false;
    this.isPaused = false;
    this.timeScale = 0.5;
    this.simTime = 0.0;
    this.routineQueue = [];
    this.currentAction = null;
    this.activeTrajectory = [];
    this.trajectoryIndex = 0;
    this.pathHistory = [];
    this.plannedSplineVisual = [];

    this.intakeVoltage = 0;
    this.rumbleActive = false;
    this.rumbleRemaining = 0;
    this.contactSeconds = 0;
    this.controllerLcdLines = ["Ready for Action", "Physics: approximate", "Press Play or [L1]"];
    this.brainLcdLines = [
      "EKF X:   0.0 in | Odom:   0.0",
      "EKF Y:   0.0 in | Odom:   0.0",
      "EKF Th:  0.0 deg",
      "Drive: HOLONOMIC X-DRIVE"
    ];

    this.telemetry = {
      time: [], vActual: [], leftVolt: [], rightVolt: [], slipAlert: false
    };

    // Official V5RC Override match state and field inventory.
    const Game=options.Game ?? globalThis.OverrideGame ?? VexRobotSimulator.dependencies?.Game;
    this.override = typeof Game === "function" ? new Game() : null;
    this.initFieldElements();
  }

  initFieldElements() {
    if (!this.override) {
      // Node physics tests load simulator.js without the browser rules bundle.
      this.mobileGoals = [
        { x: 0, y: 48, theta: 0, vx: 0, vy: 0, color: "neutral" },
        { x: -24, y: 24, theta: 0, vx: 0, vy: 0, color: "red" },
        { x: 24, y: 24, theta: 0, vx: 0, vy: 0, color: "blue" },
        { x: 0, y: -48, theta: 0, vx: 0, vy: 0, color: "neutral" },
        { x: -48, y: 0, theta: 0, vx: 0, vy: 0, color: "red" },
        { x: 48, y: 0, theta: 0, vx: 0, vy: 0, color: "blue" }
      ];
      this.rings = [
        { x: -12, y: 24, color: "red" }, { x: 12, y: 24, color: "blue" },
        { x: -36, y: 48, color: "red" }, { x: 36, y: 48, color: "blue" }
      ];
      this.cups = [];
      this.toggles = [];
      this.loaders = [];
      return;
    }
    this.mobileGoals = this.override.goals.map(goal => ({
      x: goal.x, y: goal.y, theta: 0, vx: 0, vy: 0,
      color: goal.alliance || "neutral", id: goal.id, height: goal.height, stack: goal.stack
    }));
    this.rings = this.override.pins.map(pin => ({
      x: pin.x, y: pin.y, color: pin.color, id: pin.id, kind: pin.kind
    }));
    this.cups = this.override.cups.map(cup => ({
      x: cup.x, y: cup.y, color: cup.kind === "transparent" ? "clear" : "gray",
      id: cup.id, kind: cup.kind
    }));
    this.toggles = this.override.toggles;
    this.loaders = this.override.loaders;
  }

  setPose(x, y, thetaDeg) {
    this.commandedWheelVoltages=null; this.odomVelocity=[0,0,0];
    this.productionControl?.reset();
    this.x = x;
    this.y = y;
    this.theta = thetaDeg;
    this.Vx = 0.0;
    this.Vy = 0.0;
    this.vx = 0.0;
    this.vy = 0.0;
    this.v = 0.0;
    this.w = 0.0;
    this.Ax = 0.0;
    this.Ay = 0.0;
    this.ax = 0.0;
    this.ay = 0.0;
    this.alpha = 0.0;
    this.wheelOmega = [0.0, 0.0, 0.0, 0.0];
    this.motorVolts = [0.0, 0.0, 0.0, 0.0];
    this.motorCurrents = [0.0, 0.0, 0.0, 0.0];
    this.motorTorques = [0.0, 0.0, 0.0, 0.0];
    this.wheelTractionForces = [0.0, 0.0, 0.0, 0.0];
    this.batteryVoltage = 12.6;
    this.wheelSlips = [0.0, 0.0, 0.0, 0.0];
    this.odom = { x, y, theta: thetaDeg };
    this.gyroBias = 0.0;
    this.clampedGoalIndex = -1;
    this.isPneumaticClamped = false;
    this.ekf.reset({ x, y, theta: thetaDeg });
    this.ekfPose = { x, y, theta: thetaDeg };
    this.pathHistory = [{ x, y, v: 0 }];
  }

  resetSimulation() {
    this.isRunning = false;
    this.isPaused = false;
    this.matchMode=false;this.fleet=null;
    this.managedGame=false;
    this.actionStartTime=0;
    this.actionThrottle=this.actionStrafe=this.actionTurn=0;
    this.intakeVoltage=0;
    this.rumbleActive=false;
    this.rumbleRemaining=0;
    this.contactSeconds=0;
    this.plannedSplineVisual=[];
    this.simTime=0; this.lastMotionResult="Idle";
    this.manualThrottle=this.manualStrafe=this.manualTurn=0;
    this.routineQueue = [];
    this.currentAction = null;
    this.activeTrajectory = [];
    this.trajectoryIndex = 0;
    this.setPose(0, 0, 0);
    if (this.override) this.override.reset();
    this.initFieldElements();
    this.telemetry = { time: [], vActual: [], leftVolt: [], rightVolt: [], slipAlert: false };
    this.controllerLcdLines = ["Reset Complete", "Pose: (0, 0, 0 deg)", "Physics: Ready"];
    this.updateLCDDisplays();
  }

  // Authentic 3-DOF Rigid Body & DC Motor Physics Step
  stepHolonomicPhysics(throttle_volts, strafe_volts, turn_volts, dt = 0.01, isInternalSubstep = false) {
    // The tire shear spring is deliberately stiff.  Integrating it at the
    // public 100 Hz simulation rate makes an explicit Euler solver alternate
    // between positive and negative traction every frame.  Keep the visible
    // rate at 100 Hz, but solve contact, motor, and rigid-body dynamics at
    // 2 kHz so a steady holonomic command produces a smooth trajectory.
    const MAX_CONTACT_STEP = 0.0005;
    if (!isInternalSubstep && dt > MAX_CONTACT_STEP) {
      const substepCount = Math.ceil(dt / MAX_CONTACT_STEP);
      const substepDt = dt / substepCount;
      for (let step = 0; step < substepCount; step++) {
        this.stepHolonomicPhysics(throttle_volts, strafe_volts, turn_volts, substepDt, true);
      }
      return;
    }

    // 1. Compute target terminal voltages for 45 deg X-Drive layout
    // 0: FL, 1: BL, 2: FR, 3: BR
    let targetV = [
      throttle_volts + strafe_volts + turn_volts,
      throttle_volts - strafe_volts + turn_volts,
      throttle_volts - strafe_volts - turn_volts,
      throttle_volts + strafe_volts - turn_volts
    ];

    if(this.commandedWheelVoltages) targetV=[...this.commandedWheelVoltages];
    // Scale if any motor demands > 12.0 V
    const maxDemanded = Math.max(Math.abs(targetV[0]), Math.abs(targetV[1]), Math.abs(targetV[2]), Math.abs(targetV[3]), 12.0);
    if (maxDemanded > 12.0) {
      const scale = 12.0 / maxDemanded;
      targetV = targetV.map(v => v * scale);
    }

    // Apply H-Bridge Slew Rate Limit (matches LemLib / PROS slew rate limiter)
    const maxSlewStep = this.maxVoltageSlewPerSec * dt;
    for (let i = 0; i < 4; i++) {
      this.motorVolts[i] += clamp(targetV[i] - this.motorVolts[i], -maxSlewStep, maxSlewStep);
    }

    // 2. Battery Voltage Sag Calculation
    let totalEstCurrent = 0;
    for (let i = 0; i < 4; i++) {
      const v_bemf = this.motorKe * this.wheelOmega[i];
      const i_arm = clamp((this.motorVolts[i] - v_bemf) / this.motorR, -this.motorCurrentLimit, this.motorCurrentLimit);
      totalEstCurrent += Math.abs(i_arm);
    }
    this.batteryVoltage = Math.max(10.2, 12.6 - (totalEstCurrent * this.batteryInternalR));

    // 3. DC Motor Electromechanical Torque Calculation
    for (let i = 0; i < 4; i++) {
      const effVolt = this.motorVolts[i] * (this.batteryVoltage / 12.6);
      const v_bemf = this.motorKe * this.wheelOmega[i];
      let current = (effVolt - v_bemf) / this.motorR;
      current = clamp(current, -this.motorCurrentLimit, this.motorCurrentLimit);
      this.motorCurrents[i] = current;
      this.motorTorques[i] = this.motorKt * current;
    }

    // 4. Ground Contact Velocity at each 45 deg omni-wheel
    const thetaRad = this.theta * DEG_TO_RAD;
    const cosT = Math.cos(thetaRad);
    const sinT = Math.sin(thetaRad);

    // Body velocities projected directly from conserved field velocities
    this.vx = this.Vx * cosT - this.Vy * sinT;
    this.vy = this.Vx * sinT + this.Vy * cosT;
    this.v = Math.hypot(this.Vx, this.Vy);

    const invSqrt2 = 1.0 / Math.SQRT2;
    const v_contact = [
      (this.vy + this.vx) * invSqrt2 + (this.w * this.rEff),
      (this.vy - this.vx) * invSqrt2 + (this.w * this.rEff),
      (this.vy - this.vx) * invSqrt2 - (this.w * this.rEff),
      (this.vy + this.vx) * invSqrt2 - (this.w * this.rEff)
    ];

    // Total robot mass & MOI (adjusting for clamped Mobile Goal)
    let totalMass = this.massKg;
    let totalMOI = this.moiKgM2;
    let goalDragTorque = 0.0;
    if (this.clampedGoalIndex >= 0) {
      const goalMass = 1.55; // 1.55 kg Mobile Goal
      const clampDist = 0.22; // 8.66 inches clamp distance behind robot COM
      totalMass += goalMass;
      totalMOI += goalMass * (clampDist * clampDist) + 0.008; // Parallel axis theorem + cylinder inertia
      goalDragTorque = (0.55 * Math.sign(this.w || 0)) + (0.45 * this.w); // Ground drag torque of clamped goal
    }

    const normalLoadPerWheel = (totalMass * 9.81) / 4.0;
    const maxTractionPerWheel = this.muLong * normalLoadPerWheel; // Coulomb friction limit

    // 5. Unconditionally Stable Backward-Euler Wheel Traction Integration
    const I_coupled = this.wheelInertia; // chassis mass is integrated separately
    let hasSlipAlert = false;
    for (let i = 0; i < 4; i++) {
      const num = this.wheelOmega[i] + ((this.motorTorques[i] + this.kSlip * this.wheelRadiusM * v_contact[i]) * dt) / I_coupled;
      const den = 1.0 + ((this.kSlip * this.wheelRadiusM * this.wheelRadiusM * dt) / I_coupled);
      const previousOmega=this.wheelOmega[i];
      this.wheelOmega[i] = num / den;

      const v_wheel = this.wheelOmega[i] * this.wheelRadiusM;
      const slipVel = v_wheel - v_contact[i];
      this.wheelSlips[i] = slipVel;

      if (Math.abs(slipVel) > 0.25 && Math.abs(v_contact[i]) > 0.1) {
        hasSlipAlert = true;
      }

      const f_ideal = this.kSlip * slipVel;
      this.wheelTractionForces[i] = clamp(f_ideal, -maxTractionPerWheel, maxTractionPerWheel);
      if(Math.abs(f_ideal)>maxTractionPerWheel){
        this.wheelOmega[i]=previousOmega+(this.motorTorques[i]-this.wheelTractionForces[i]*this.wheelRadiusM)*dt/I_coupled;
        this.wheelSlips[i]=this.wheelOmega[i]*this.wheelRadiusM-v_contact[i];
      }
    }
    this.telemetry.slipAlert = hasSlipAlert;

    // 6. Net Body Forces & Moments in Robot Coordinate Frame
    const F0 = this.wheelTractionForces[0];
    const F1 = this.wheelTractionForces[1];
    const F2 = this.wheelTractionForces[2];
    const F3 = this.wheelTractionForces[3];

    // Rolling resistance & aerodynamic drag
    const rollFrictionY = (Math.abs(this.vy) > 0.001) ? (this.muLat * normalLoadPerWheel * Math.sign(this.vy) + 1.4 * this.vy) : 0.0;
    const rollFrictionX = (Math.abs(this.vx) > 0.001) ? (this.muLat * normalLoadPerWheel * Math.sign(this.vx) + 1.4 * this.vx) : 0.0;
    const rotDrag = (Math.abs(this.w) > 0.001) ? (0.85 * this.w + 0.25 * Math.sign(this.w)) : 0.0;

    const F_y_body = (F0 + F1 + F2 + F3) * invSqrt2 - rollFrictionY;
    const F_x_body = (F0 - F1 - F2 + F3) * invSqrt2 - rollFrictionX;
    const Tau_yaw = ((F0 + F1 - F2 - F3) * this.rEff) - rotDrag - goalDragTorque;

    // 7. Rigid Body Newton-Euler Accelerations in Inertial Field Frame
    const F_field_X = F_x_body * cosT + F_y_body * sinT;
    const F_field_Y = -F_x_body * sinT + F_y_body * cosT;

    this.Ax = F_field_X / totalMass;
    this.Ay = F_field_Y / totalMass;
    this.alpha = Tau_yaw / totalMOI;

    // Direct inertial integration (zero Coriolis drift, zero spiral instability)
    this.Vx += this.Ax * dt;
    this.Vy += this.Ay * dt;
    this.w += this.alpha * dt;

    // Static friction lock when unpowered and stationary
    const isStationaryLinear = Math.hypot(this.Vx, this.Vy) < 0.015;
    const isStationaryRot = Math.abs(this.w) < 0.015;
    const isUnpowered = targetV.every(v=>Math.abs(v)<0.05);

    if (isUnpowered && isStationaryLinear) {
      this.Vx = 0.0;
      this.Vy = 0.0;
    }
    if (isUnpowered && isStationaryRot) {
      this.w = 0.0;
    }

    // Body accelerations for telemetry
    this.ax = this.Ax * cosT - this.Ay * sinT;
    this.ay = this.Ax * sinT + this.Ay * cosT;

    // 8. Integrate Global Ground-Truth Pose
    const dx_m = this.Vx * dt;
    const dy_m = this.Vy * dt;
    const dTheta_rad = this.w * dt;

    this.x += dx_m * METER_TO_INCH;
    this.y += dy_m * METER_TO_INCH;
    this.theta = normalizeAngle(thetaRad + dTheta_rad) * RAD_TO_DEG;

    // Legacy fallback for stand-alone plant tests without Override geometry.
    if(!this.override){
      const fieldLimit = 70.2 - this.trackWidthInches / 2.0;
      if (this.x > fieldLimit) { this.x = fieldLimit; if (this.Vx > 0) this.Vx *= -0.15; }
      if (this.x < -fieldLimit) { this.x = -fieldLimit; if (this.Vx < 0) this.Vx *= -0.15; }
      if (this.y > fieldLimit) { this.y = fieldLimit; if (this.Vy > 0) this.Vy *= -0.15; }
      if (this.y < -fieldLimit) { this.y = -fieldLimit; if (this.Vy < 0) this.Vy *= -0.15; }
    }

    // Encoder odometry matches the four-wheel fallback fitted in main.cpp.
    // Slip corrupts this estimate; ground-truth translation is never a sensor.
    const rim=this.wheelOmega.map(w=>w*this.wheelRadiusM);
    const sBody=(rim[0]-rim[1]-rim[2]+rim[3])/(2*Math.SQRT2);
    const fBody=(rim[0]+rim[1]+rim[2]+rim[3])/(2*Math.SQRT2);
    const yaw=(rim[0]+rim[1]-rim[2]-rim[3])/(4*this.rEff);
    const sensorHeading=this.odom.theta*DEG_TO_RAD+yaw*dt/2;
    const sx=sBody*Math.cos(sensorHeading)+fBody*Math.sin(sensorHeading);
    const sy=-sBody*Math.sin(sensorHeading)+fBody*Math.cos(sensorHeading);
    this.odomVelocity=[sx,sy,yaw];
    this.odom.x+=sx*dt*METER_TO_INCH;
    this.odom.y+=sy*dt*METER_TO_INCH;
    this.odom.theta=normalizeAngle(this.odom.theta*DEG_TO_RAD+yaw*dt)*RAD_TO_DEG;

    // 10. Extended Kalman Filter (EKF) Sensor Fusion
    this.ekf.predict(dt);
    this.ekf.updatePose(
      this.odom.x * INCH_TO_METER,
      this.odom.y * INCH_TO_METER,
      this.odom.theta * DEG_TO_RAD,
      0.015, 0.008
    );
    this.ekfPose = this.ekf.getPose();

    // 11. Mobile Goals Physical Interaction
    this.updateMobileGoalsPhysics(dt);

    // Trail history
    if (this.pathHistory.length === 0 ||
        Math.hypot(this.x - this.pathHistory[this.pathHistory.length - 1].x,
                   this.y - this.pathHistory[this.pathHistory.length - 1].y) > 0.4) {
      this.pathHistory.push({ x: this.x, y: this.y, v: Math.abs(this.v) });
      if (this.pathHistory.length > 800) this.pathHistory.shift();
    }
  }

  updateMobileGoalsPhysics(dt) {
    // Override goals are fixed field elements; scoring and placement live in OverrideGame.
    if (this.override){
      // Rotated body contacts affect truth/velocity, never encoder feedback.
      const pose={x:this.x,y:this.y,theta:this.theta};
      if(!this.matchMode){pose.width=this.trackWidthInches;pose.length=this.trackWidthInches;}
      const contact=this.override.resolveRobotContact(this.activeRobotId,pose,this.matchMode);
      if(contact.normals.length)this.contactSeconds+=dt;
      this.x=contact.x;this.y=contact.y;
      for(const n of contact.normals){
        const inward=this.Vx*n.x+this.Vy*n.y;
        if(inward<0){this.Vx-=inward*n.x;this.Vy-=inward*n.y;}
      }
      return;
    }
    const rad = this.theta * DEG_TO_RAD;
    const clampWorldX = this.x - 7.5 * Math.sin(rad);
    const clampWorldY = this.y - 7.5 * Math.cos(rad);

    for (let i = 0; i < this.mobileGoals.length; i++) {
      const goal = this.mobileGoals[i];

      if (this.clampedGoalIndex === i) {
        goal.x = clampWorldX;
        goal.y = clampWorldY;
        goal.theta = this.theta;
        goal.vx = this.vx;
        goal.vy = this.vy;
        continue;
      }

      if (this.isPneumaticClamped && this.clampedGoalIndex === -1) {
        const distToClamp = Math.hypot(goal.x - clampWorldX, goal.y - clampWorldY);
        if (distToClamp < 4.5) {
          this.clampedGoalIndex = i;
          this.triggerRumble("..");
          this.controllerLcdLines[0] = "Goal Clamped! (+1.5kg)";
          continue;
        }
      }

      const distToCenter = Math.hypot(goal.x - this.x, goal.y - this.y);
      const minDist = 11.0;
      if (distToCenter < minDist) {
        const overlap = minDist - distToCenter;
        const pushDirX = (goal.x - this.x) / (distToCenter || 1.0);
        const pushDirY = (goal.y - this.y) / (distToCenter || 1.0);

        goal.x += pushDirX * overlap * 0.7;
        goal.y += pushDirY * overlap * 0.7;

        goal.vx = (goal.vx || 0) + pushDirX * (this.v * 0.8);
        goal.vy = (goal.vy || 0) + pushDirY * (this.v * 0.8);
      }

      if (goal.vx || goal.vy) {
        goal.x += (goal.vx || 0) * dt * METER_TO_INCH;
        goal.y += (goal.vy || 0) * dt * METER_TO_INCH;
        goal.vx *= Math.max(0, 1.0 - 5.0 * dt);
        goal.vy *= Math.max(0, 1.0 - 5.0 * dt);
      }
    }
  }

  queueAction(action) {
    this.routineQueue.push(action);
  }

  startRoutine(name) {
    this.resetSimulation();
    if(!this.productionControl){this.lastMotionResult="ControlUnavailable";this.controllerLcdLines[0]="C++ control unavailable";return;}
    this.isRunning = true;

    switch (name) {
      case "autoHolonomicSkills":
        this.buildHolonomicSkillsRoutine();
        break;
      case "autoHolonomicRedAWP":
        this.buildHolonomicRedAWPRoutine();
        break;
      case "autoHolonomicGoalRush":
        this.buildHolonomicGoalRushRoutine();
        break;
      case "autoSkills":
        this.buildSkillsRoutine();
        break;
      case "autoRedAWP":
        this.buildRedAWPRoutine();
        break;
      case "autoBlueAWP":
        this.buildBlueAWPRoutine();
        break;
      case "autoHybridDemo":
        this.buildHybridDemoRoutine();
        break;
      case "testTrackWidth":
        this.buildTrackWidthTest();
        break;
      case "testLinearDrive":
        this.buildLinearDriveTest(24.0);
        break;
      case "testAngularTurn":
        this.buildAngularTurnTest(90.0);
        break;
      case "competitionAutonomous":
        this.queueAction({type:"spline",end:{x:24,y:24,theta:90},maxVel:.45,maxAccel:.8});
        this.queueAction({type:"pose",targetX:0,targetY:0,targetTheta:0});break;
      case "testStrafe":this.queueAction({type:"strafe",targetInches:24,heading:0});break;
      case "testDiagonal":this.queueAction({type:"pose",targetX:24,targetY:24,targetTheta:0});break;
      case "testBezier":
        this.queueAction({type:"bezier",points:[[0,0],[0,16],[24,8],[24,24]],startHeading:0,endHeading:90,desc:"Pedro Bezier -> LTV-LQR -> PID"});break;
      case "testQuinticSpline":
        this.buildSplineTest();
        break;
      case "testFeedforward":
        this.buildFeedforwardTest();
        break;
      default:
        this.buildHolonomicSkillsRoutine();
    }
    this.controllerLcdLines[0] = `Auto: ${name}`;
  }

  // 3-DOF Holonomic Skills Routine
  buildHolonomicSkillsRoutine() {
    this.queueAction({ type: "intake", voltage: 12000, desc: "Intake ON" });
    this.queueAction({ type: "drive", targetInches: 24, heading: 0, desc: "[Holo 1] Forward 24in to (0, 24)" });
    this.queueAction({ type: "strafe", targetInches: 24, heading: 0, desc: "[Holo 2] Sideways Strafe Right 24in" });
    this.queueAction({ type: "diagonal", targetX: 0, targetY: 24, endHeading: 0, desc: "[Holo 3] Strafe Return to (0, 24)" });
    this.queueAction({ type: "drive", targetInches: -24, heading: 0, desc: "[Holo 4] Reverse 24in to (0, 0)" });
    this.queueAction({
      type: "spline",
      start: { x: 0, y: 0, theta: 0 },
      end: { x: 24, y: 24, theta: 90 },
      maxVel: 1.1, maxAccel: 1.8,
      desc: "[Holo 5] Quintic Spline to (24, 24)"
    });
    this.queueAction({
      type: "pose",
      targetX: 0, targetY: 0, targetTheta: 0, desc: "[Holo 6] Boomerang curve to (0, 0)"
    });
    this.queueAction({ type: "intake", voltage: 0, desc: "Intake Stop" });
  }

  buildHolonomicRedAWPRoutine() {
    this.queueAction({ type: "intake", voltage: 12000, desc: "Intake ON" });
    this.queueAction({ type: "drive", targetInches: 14, heading: 0, desc: "Score Alliance Stake" });
    this.queueAction({ type: "strafe", targetInches: -16, heading: 0, desc: "Lateral Strafe Left" });
    this.queueAction({ type: "diagonal", forwardInches: 18, strafeInches: -8, endHeading: -45, desc: "Diagonal Dash into Goal" });
    this.queueAction({ type: "clamp", clamp: true, desc: "Pneumatic Clamp ON" });
    this.queueAction({
      type: "spline",
      start: { x: -24, y: 24, theta: -45 },
      end: { x: -36, y: 48, theta: 0 },
      maxVel: 1.0, maxAccel: 1.8,
      desc: "LTV Spline with Clamped Goal"
    });
    this.queueAction({ type: "intake", voltage: 0, desc: "Intake Stop" });
  }

  buildHolonomicGoalRushRoutine() {
    this.queueAction({ type: "intake", voltage: 12000, desc: "Intake ON" });
    this.queueAction({ type: "diagonal", forwardInches: 48, strafeInches: 12, endHeading: 15, desc: "Diagonal Rush Goal" });
    this.queueAction({ type: "clamp", clamp: true, desc: "Clamp Mobile Goal" });
    this.queueAction({ type: "diagonal", forwardInches: -36, strafeInches: -12, endHeading: 0, desc: "Reverse Strafe Pull Goal" });
    this.queueAction({ type: "intake", voltage: 0, desc: "Intake Stop" });
  }

  buildSkillsRoutine() {
    this.queueAction({ type: "drive", targetInches: 24, heading: 0, desc: "Straight 24in to (0, 24)" });
    this.queueAction({ type: "turn", targetHeading: 90, desc: "Turn Right 90 deg" });
    this.queueAction({ type: "drive", targetInches: 24, heading: 90, desc: "Straight 24in to (24, 24)" });
    this.queueAction({ type: "drive", targetInches: -24, heading: 90, desc: "Back 24in to (0, 24)" });
    this.queueAction({ type: "turn", targetHeading: 0, desc: "Turn Left 90 deg" });
    this.queueAction({ type: "drive", targetInches: -24, heading: 0, desc: "Back 24in to (0, 0)" });
    this.queueAction({
      type: "spline",
      start: { x: 0, y: 0, theta: 0 },
      end: { x: 24, y: 24, theta: 90 },
      maxVel: 1.0, maxAccel: 1.8,
      desc: "LTV Quintic Spline to (24, 24)"
    });
    this.queueAction({
      type: "pose",
      targetX: 0, targetY: 0, targetTheta: 0, desc: "Boomerang Curve back to (0, 0)"
    });
  }

  buildRedAWPRoutine() {
    this.queueAction({ type: "intake", voltage: 12000, desc: "Intake ON" });
    this.queueAction({ type: "drive", targetInches: 14, heading: 0, desc: "Score Alliance Stake" });
    this.queueAction({
      type: "spline",
      start: { x: 0, y: 14, theta: 0 },
      end: { x: -16, y: 32, theta: -45 },
      maxVel: 1.1, maxAccel: 1.8,
      desc: "LTV S-Curve to Mobile Goal"
    });
    this.queueAction({ type: "turn", targetHeading: -135, desc: "Snap Turn to Goal" });
    this.queueAction({ type: "drivePoint", targetX: -24, targetY: 24, desc: "Clamp Mobile Goal" });
    this.queueAction({ type: "clamp", clamp: true, desc: "Clamp Goal" });
    this.queueAction({ type: "intake", voltage: 0, desc: "Intake Stop" });
  }

  buildBlueAWPRoutine() {
    this.queueAction({ type: "intake", voltage: 12000, desc: "Intake ON" });
    this.queueAction({ type: "drive", targetInches: 14, heading: 0, desc: "Score Alliance Stake" });
    this.queueAction({
      type: "spline",
      start: { x: 0, y: 14, theta: 0 },
      end: { x: 16, y: 32, theta: 45 },
      maxVel: 1.1, maxAccel: 1.8,
      desc: "LTV S-Curve to Mobile Goal"
    });
    this.queueAction({ type: "turn", targetHeading: 135, desc: "Snap Turn to Goal" });
    this.queueAction({ type: "drivePoint", targetX: 24, targetY: 24, desc: "Clamp Mobile Goal" });
    this.queueAction({ type: "clamp", clamp: true, desc: "Clamp Goal" });
    this.queueAction({ type: "intake", voltage: 0, desc: "Intake Stop" });
  }

  buildHybridDemoRoutine() {
    this.queueAction({
      type: "spline",
      start: { x: 0, y: 0, theta: 0 },
      end: { x: 18.0, y: 36.0, theta: 45.0 },
      maxVel: 1.0, maxAccel: 1.8,
      desc: "LTV Quintic Spline"
    });
    this.queueAction({ type: "turn", targetHeading: 90.0, desc: "LQR Snap Turn (90 deg)" });
    this.queueAction({ type: "drivePoint", targetX: 30.0, targetY: 36.0, desc: "LQR Drive to (30, 36)" });
    this.queueAction({ type: "turn", targetHeading: 0.0, desc: "LQR Snap Turn (0 deg)" });
  }

  buildTrackWidthTest() {
    this.queueAction({ type: "turn", targetHeading: 180.0, desc: "Spin: 0 to 180 deg" });
    this.queueAction({ type: "turn", targetHeading: 0.0, desc: "Turn: return to 0 deg" });
  }

  buildLinearDriveTest(inches = 24.0) {
    this.queueAction({ type: "drivePoint", targetX: 0, targetY: inches, desc: `Test: Lin ${inches}in` });
  }

  buildAngularTurnTest(heading = 90.0) {
    this.queueAction({ type: "turn", targetHeading: heading, desc: `Test: Turn ${heading}deg` });
  }

  buildSplineTest() {
    this.queueAction({
      type: "spline",
      start: { x: 0, y: 0, theta: 0 },
      end: { x: 20.0, y: 40.0, theta: 45.0 },
      maxVel: 1.0, maxAccel: 1.8,
      desc: "Gen Spline 5th..."
    });
  }

  buildFeedforwardTest() {
    this.queueAction({ type: "feedforward", desc: "Calibrating kS..." });
  }

  startGame(mode,choices={}) {
    if(!this.productionControl)return {ok:false,error:'C++ controller unavailable'};
    this.resetSimulation();this.matchMode=true;this.override.startMatch(mode);
    this.fleet=new this.Fleet(this,VexRobotSimulator,choices);this.fleet.syncView();this.isPaused=false;
    return {ok:true};
  }

  // Update Loop
  update(dt = 0.01) {
    if (this.isPaused) return;
    if(this.fleet){this.fleet.update(dt);this.updateRumble(dt);return;}
    if(!Number.isFinite(dt)||dt<=0||dt>0.1){this.isRunning=false;this.currentAction=null;this.routineQueue=[];this.commandedWheelVoltages=null;this.motorVolts=[0,0,0,0];this.lastMotionResult="InvalidDt";return;}

    let throttle_v = this.manualThrottle || 0.0;
    let strafe_v = this.manualStrafe || 0.0;
    let turn_v = this.manualTurn || 0.0;
    if(this.matchMode&&(this.override.phase!=="driver"))throttle_v=strafe_v=turn_v=0;

    if (this.isRunning && this.currentAction == null && this.routineQueue.length > 0) {
      this.currentAction = this.routineQueue.shift();
      this.actionStartTime = this.simTime;
      this.initAction(this.currentAction);
    }

    if (this.isRunning && this.currentAction) {
      const isDone = this.executeAction(this.currentAction, dt);
      throttle_v = this.actionThrottle || 0.0;
      strafe_v = this.actionStrafe || 0.0;
      turn_v = this.actionTurn || 0.0;

      if (isDone) {
        this.currentAction = null;
        this.actionThrottle = 0.0;
        this.actionStrafe = 0.0;
        this.actionTurn = 0.0;
        throttle_v = 0.0;
        strafe_v = 0.0;
        turn_v = 0.0;
        if (this.routineQueue.length === 0) {
          this.isRunning = false;
          this.triggerRumble("..");
          this.controllerLcdLines[0] = this.lastMotionResult==="Settled"?"Auto Complete":"Auto Stopped";
          this.controllerLcdLines[1] = this.lastMotionResult;
        }
      }
    }

    if(!this.isRunning || !this.currentAction) this.commandedWheelVoltages=null;
    this.stepHolonomicPhysics(throttle_v, strafe_v, turn_v, dt);
    if (this.override && this.matchMode) {
      this.override.setRobotPose(this.activeRobotId, {
        x: this.x, y: this.y, theta: this.theta
      });
      const robot=this.override.robots.find(r=>r.id===this.activeRobotId);
      robot.velocity={vx:this.Vx*METER_TO_INCH,vy:this.Vy*METER_TO_INCH,omega:this.w};
      if(!this.managedGame)this.override.tick(dt);
      this.override.goals.forEach((goal, i) => {
        if (this.mobileGoals[i]) {
          this.mobileGoals[i].x = goal.x;
          this.mobileGoals[i].y = goal.y;
          this.mobileGoals[i].stack = goal.stack;
        }
      });
    }
    this.simTime += dt;
    this.updateRumble(dt);
    this.updateTelemetry(dt, throttle_v, turn_v);
    this.updateLCDDisplays();
  }

  initAction(action) {
    if (action.type === "policyReference") {
      action.duration = 0;
      this.productionControl?.reset();
      this.lastMotionResult = "Running";
      return;
    }
    if (action.desc) {
      this.controllerLcdLines[0] = action.desc.substring(0, 18);
    }

    if (action.type === "spline") {
      const start = { x: this.odom.x, y: this.odom.y, theta: this.odom.theta };
      this.activeTrajectory = this.productionControl?.spline(
        start, action.end, Math.min(action.maxVel || 0.45,0.45), Math.min(action.maxAccel || 0.8,0.8), 3.5, 0.01
      ) || [];
      this.trajectoryIndex = 0;
      this.plannedSplineVisual = this.activeTrajectory.map(s => ({ x: s.x * METER_TO_INCH, y: s.y * METER_TO_INCH }));
    } else if (action.type === "drive") {
      action.initialHeading = (action.heading !== undefined) ? action.heading : this.odom.theta;
      action.targetHeading = action.initialHeading;
      const rad = action.initialHeading * DEG_TO_RAD;
      action.startX = this.odom.x;
      action.startY = this.odom.y;
      action.targetX = this.odom.x + action.targetInches * Math.sin(rad);
      action.targetY = this.odom.y + action.targetInches * Math.cos(rad);
    } else if (action.type === "strafe") {
      action.initialHeading = (action.heading !== undefined) ? action.heading : this.odom.theta;
      action.targetHeading = action.initialHeading;
      const rad = action.initialHeading * DEG_TO_RAD;
      action.targetX = this.odom.x + action.targetInches * Math.cos(rad);
      action.targetY = this.odom.y - action.targetInches * Math.sin(rad);
    } else if (action.type === "diagonal") {
      action.initialHeading = this.odom.theta;
      const rad = action.initialHeading * DEG_TO_RAD;
      if (action.targetX !== undefined && action.targetY !== undefined) {
        // Absolute field waypoint target
      } else {
        action.targetX = this.odom.x + (action.strafeInches || 0) * Math.cos(rad) + (action.forwardInches || 0) * Math.sin(rad);
        action.targetY = this.odom.y - (action.strafeInches || 0) * Math.sin(rad) + (action.forwardInches || 0) * Math.cos(rad);
      }
    } else if (action.type === "drivePoint" || action.type === "pose") {
      action.initialHeading = this.odom.theta;
    } else if (action.type === "turn") {
      action.initialHeading = this.odom.theta;
    } else if (action.type === "intake") {
      this.intakeVoltage = action.voltage;
    }
    if(["drive","strafe","diagonal","drivePoint","pose","turn","spline","bezier"].includes(action.type)){
      action.startSI=[this.odom.x*INCH_TO_METER,this.odom.y*INCH_TO_METER,this.odom.theta*DEG_TO_RAD];
      action.endSI=[(action.targetX??this.odom.x)*INCH_TO_METER,(action.targetY??this.odom.y)*INCH_TO_METER,
        (action.targetHeading??action.targetTheta??action.endHeading??this.odom.theta)*DEG_TO_RAD];
      action.duration=action.type==="spline"?(this.activeTrajectory.at(-1)?.time??-1):this.productionControl?.duration(action.startSI,action.endSI);
      if(action.type==="bezier")action.duration=this.productionControl?.bezier(action.points,action.startHeading*DEG_TO_RAD,action.endHeading*DEG_TO_RAD);
      action.settledSeconds=0;
      // Defaults follow the feasible reference; explicit caller deadlines are honored.
      if(action.timeout == null || action.timeout === 0) action.timeout=(action.duration+3)*1000;
      this.productionControl?.reset();
      this.lastMotionResult="Running";
    }
  }

  executeAction(action, dt) {
    const elapsedMs = (this.simTime - this.actionStartTime) * 1000;

    if (action.type === "intake") return true;

    if (action.type === "clamp") {
      this.isPneumaticClamped = action.clamp;
      if (!this.isPneumaticClamped && this.clampedGoalIndex >= 0) {
        this.clampedGoalIndex = -1;
        this.controllerLcdLines[0] = "Goal Released";
      }
      return true;
    }

    const fail=(result)=>{this.lastMotionResult=result;this.routineQueue=[];this.commandedWheelVoltages=[0,0,0,0];return true;};
    if(!this.productionControl || !Number.isFinite(action.duration) || action.duration<0) return fail("InvalidInput");
    const time=elapsedMs/1000;
    if(!Number.isFinite(action.timeout)||action.timeout<=0||action.timeout>120000)return fail("InvalidInput");
    if(elapsedMs>=action.timeout) return fail("TimedOut");
    let ref;
    if(action.type==="policyReference"){
      if(!Array.isArray(action.referenceSI)||action.referenceSI.length!==6||!action.referenceSI.every(Number.isFinite))return fail("InvalidInput");
      ref=action.referenceSI;
    }else if(action.type==="spline"){
      while(this.trajectoryIndex+1<this.activeTrajectory.length&&this.activeTrajectory[this.trajectoryIndex+1].time<=time)this.trajectoryIndex++;
      const a=this.activeTrajectory[this.trajectoryIndex],b=this.activeTrajectory[this.trajectoryIndex+1];
      let x=a.x,y=a.y,h=a.heading,v=a.linear_vel,w=a.angular_vel;
      if(b){const q=clamp((time-a.time)/(b.time-a.time),0,1);x+=(b.x-x)*q;y+=(b.y-y)*q;h+=normalizeAngle(b.heading-h)*q;v+=(b.linear_vel-v)*q;w+=(b.angular_vel-w)*q;}
      if(time>=action.duration){v=0;w=0;}
      ref=[x,y,Math.PI/2-h,v*Math.cos(h),v*Math.sin(h),-w];
    }else if(action.type==="bezier")ref=this.productionControl.bezierReference(time);
    else ref=this.productionControl.reference(action.startSI,action.endSI,time,action.duration);
    const feedback=[this.odom.x*INCH_TO_METER,this.odom.y*INCH_TO_METER,this.odom.theta*DEG_TO_RAD,
      ...this.odomVelocity,...this.wheelOmega.map(w=>w*this.wheelRadiusM)];
    const output=this.productionControl.step(ref,feedback,dt);
    if(!output.valid) return fail("SensorFault");
    this.commandedWheelVoltages=output.volts;
    if(action.type==="policyReference")return false;
    const close=Math.hypot(ref[0]-feedback[0],ref[1]-feedback[1])<0.02032&&Math.abs(normalizeAngle(ref[2]-feedback[2]))<0.035&&Math.hypot(feedback[3],feedback[4])<0.0254&&Math.abs(feedback[5])<0.0873;
    action.settledSeconds=time>=action.duration&&close?action.settledSeconds+dt:0;
    if(action.settledSeconds>=0.15){this.lastMotionResult="Settled";return true;}
    return false;
  }

  // Manual 3-DOF Holonomic Arcade Drive from Keyboard
  holonomicArcade(forward, strafe, turn) {
    this.manualThrottle = forward * 12.0; // -12V to +12V
    this.manualStrafe = strafe * 12.0;   // -12V to +12V
    this.manualTurn = turn * 10.0;       // -10V to +10V
  }

  triggerRumble(pattern) {
    this.rumbleActive = true;
    this.rumbleRemaining = 0.3;
  }

  updateRumble(dt) {
    if(!Number.isFinite(dt)||dt<=0)return;
    this.rumbleRemaining=Math.max(0,this.rumbleRemaining-dt);
    this.rumbleActive=this.rumbleRemaining>0;
  }

  updateTelemetry(dt, throttle, turn) {
    const t = this.simTime;
    this.telemetry.time.push(t);
    this.telemetry.vActual.push(this.v);
    this.telemetry.leftVolt.push(this.motorVolts[0]);
    this.telemetry.rightVolt.push(this.motorVolts[2]);

    if (this.telemetry.time.length > 150) {
      this.telemetry.time.shift();
      this.telemetry.vActual.shift();
      this.telemetry.leftVolt.shift();
      this.telemetry.rightVolt.shift();
    }
  }

  updateLCDDisplays() {
    this.brainLcdLines[0] = `EKF X: ${this.ekfPose.x.toFixed(1).padStart(5, ' ')}" | Odom: ${this.odom.x.toFixed(1).padStart(5, ' ')}"`;
    this.brainLcdLines[1] = `EKF Y: ${this.ekfPose.y.toFixed(1).padStart(5, ' ')}" | Odom: ${this.odom.y.toFixed(1).padStart(5, ' ')}`;
    this.brainLcdLines[2] = `EKF Th: ${this.ekfPose.theta.toFixed(1).padStart(5, ' ')}° | Wheel yaw: ${this.odom.theta.toFixed(1).padStart(5, ' ')}°`;
    const massLabel = (this.clampedGoalIndex >= 0) ? "MASS: 8.35kg [GOAL]" : "MASS: 6.80kg [NORM]";
    this.brainLcdLines[3] = `${massLabel} | BATT: ${this.batteryVoltage.toFixed(1)}V`;
  }
}

const NationalsEngine = {VexRobotSimulator, Matrix, RobotEKF, LQRController, PIDController};
if (typeof module === 'object' && module.exports) {
  VexRobotSimulator.dependencies = {
    config: require('../config/robot.json'),
    Game: require('./override'),
    Fleet: require('./override-autonomy')
  };
  module.exports = NationalsEngine;
} else {
  globalThis.NationalsEngine = NationalsEngine;
}
