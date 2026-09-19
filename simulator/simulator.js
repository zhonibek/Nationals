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
  constructor(control = null) {
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

    const config=globalThis.ROBOT_CONFIG;
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
    this.override = typeof OverrideGame === "function" ? new OverrideGame() : null;
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
    this.matchMode=false;
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

  // Update Loop
  update(dt = 0.01) {
    if (this.isPaused) return;
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
      this.override.tick(dt);
      this.override.goals.forEach((goal, i) => {
        if (this.mobileGoals[i]) {
          this.mobileGoals[i].x = goal.x;
          this.mobileGoals[i].y = goal.y;
          this.mobileGoals[i].stack = goal.stack;
        }
      });
    }
    this.simTime += dt;
    this.updateTelemetry(dt, throttle_v, turn_v);
    this.updateLCDDisplays();
  }

  initAction(action) {
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
    if(action.type==="spline"){
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
    setTimeout(() => { this.rumbleActive = false; }, 300);
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

// ============================================================================
// 7. Field Canvas 2D Renderer (With X-Drive 45° Wheels & Proportional Elements)
// ============================================================================
class FieldRenderer {
  constructor(canvas, simulator) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.sim = simulator;

    this.showTargetPath = true;
    this.showTrail = true;
    this.showEKF = true;
    this.showCoordinates = true;
    this.showFieldElements = true;

    // Jerry.io Planner State
    this.jerryWaypoints = [];
    this.selectedWpIndex = -1;
    this.draggedWpIndex = -1;
    this.dragMode = 'none'; // 'pos' or 'heading'

    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
  }

  resizeCanvas() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const size = Math.min(rect.width, rect.height || rect.width);
    this.canvas.width = size * window.devicePixelRatio;
    this.canvas.height = size * window.devicePixelRatio;
    this.scale = this.canvas.width / 140.4;
  }

  toCanvas(xInches, yInches) {
    const cx = this.canvas.width / 2;
    const cy = this.canvas.height / 2;
    return {
      x: cx + (xInches * this.scale),
      y: cy - (yInches * this.scale)
    };
  }

  toField(px, py) {
    const cx = this.canvas.width / 2;
    const cy = this.canvas.height / 2;
    return {
      x: (px - cx) / this.scale,
      y: (cy - py) / this.scale
    };
  }

  render() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    this.drawFieldBackground();
    if (this.showFieldElements) this.drawFieldElements();
    if (this.showTargetPath) this.drawPlannedTrajectory();
    this.drawJerryWaypoints();
    if (this.showTrail) this.drawPathTrail();
    if (this.showEKF) this.drawEKFEllipse();
    this.drawRobot();
  }

  drawFieldBackground() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const tileSize = 23.4 * this.scale;

    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 6; c++) {
        const x = c * tileSize;
        const y = r * tileSize;
        ctx.fillStyle = (r + c) % 2 === 0 ? '#737982' : '#808791';
        ctx.fillRect(x, y, tileSize, tileSize);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, tileSize, tileSize);
      }
    }

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.65)';
    ctx.lineWidth = 3;

    // Diagonal quadrant boundaries; central diamond is drawn by OverrideView.
    for(const [a,b] of [[[-60,60],[-11.555,11.555]],[[11.555,-11.555],[60,-60]],[[-60,-60],[-11.555,-11.555]],[[11.555,11.555],[60,60]]]){
      const p=this.toCanvas(...a),q=this.toCanvas(...b);ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();
    }
    ctx.strokeStyle = '#384661';
    ctx.lineWidth = 8;
    ctx.strokeRect(4, 4, w - 8, h - 8);

    ctx.fillStyle = '#f43f5e';
    ctx.fillRect(4, 4, 28, 28);
    ctx.fillStyle = '#3b82f6';
    ctx.fillRect(w - 32, h - 32, 28, 28);

    if (this.showCoordinates) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.22)';
      ctx.font = `${Math.max(9, 10 * (this.scale / 4.5))}px JetBrains Mono`;
      for (let inX = -48; inX <= 48; inX += 24) {
        for (let inY = -48; inY <= 48; inY += 48) {
          const pt = this.toCanvas(inX, inY);
          ctx.fillText(`(${inX}", ${inY}")`, pt.x + 4, pt.y - 4);
        }
      }
    }
  }

  // Reduced, proportional game elements (User request: smaller elements)
  drawFieldElements() {
    const ctx = this.ctx;
    if(this.sim.override && typeof OverrideView!=="undefined"){OverrideView.draw2D(this);return;}

    // Center Ladder: reduced from 10 to 5.5 inches
    const center = this.toCanvas(0, 0);
    const ladderRadius = 5.5 * this.scale;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(center.x, center.y, ladderRadius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Rings: reduced from 3.5 to 1.8 inches
    for (const ring of this.sim.rings) {
      const pt = this.toCanvas(ring.x, ring.y);
      const rSize = 1.8 * this.scale;
      ctx.fillStyle = ring.color === "red" ? "#e11d48" : (ring.color === "blue" ? "#2563eb" : "#facc15");
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, rSize, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#0b0f19';
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, rSize * 0.45, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Cups are neutral scoring objects: they affect placement but carry no points.
    for (const cup of (this.sim.cups || [])) {
      const pt = this.toCanvas(cup.x, cup.y);
      const rSize = 2.1 * this.scale;
      ctx.fillStyle = cup.color === "clear" ? "rgba(180,220,255,0.25)" : "#94a3b8";
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.rect(pt.x - rSize, pt.y - rSize, rSize * 2, rSize * 2);
      ctx.fill();
      ctx.stroke();
    }

    // Override goals, with alliance-colored and neutral/tall variants.
    for (const goal of this.sim.mobileGoals) {
      const pt = this.toCanvas(goal.x, goal.y);
      const goalRadius = 3.6 * this.scale;

      ctx.fillStyle = goal.color === 'neutral' ? '#eab308' : (goal.color === 'red' ? '#be123c' : '#1d4ed8');
      ctx.strokeStyle = '#f8fafc';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, goalRadius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      // Center post
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 1.0 * this.scale, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  drawPlannedTrajectory() {
    const pts = this.sim.plannedSplineVisual;
    if (!pts || pts.length < 2) return;

    const ctx = this.ctx;
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();

    for (let i = 0; i < pts.length; i++) {
      const pt = this.toCanvas(pts[i].x, pts[i].y);
      if (i === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Jerry.io Visual Waypoints & Path Curve
  drawJerryWaypoints() {
    const wps = this.jerryWaypoints;
    if (!wps || wps.length === 0) return;

    const ctx = this.ctx;

    // Draw connecting spline/polyline path
    if (wps.length >= 2) {
      ctx.strokeStyle = '#10b981'; // Emerald path
      ctx.lineWidth = 2.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      for (let i = 0; i < wps.length; i++) {
        const pt = this.toCanvas(wps[i].x, wps[i].y);
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw waypoints
    for (let i = 0; i < wps.length; i++) {
      const wp = wps[i];
      const pt = this.toCanvas(wp.x, wp.y);
      const isSelected = (i === this.selectedWpIndex);

      // Pin circle
      ctx.fillStyle = isSelected ? '#3b82f6' : '#10b981';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 11, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      // Number
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px Outfit';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(i + 1, pt.x, pt.y);

      // Orientation Heading Needle
      const headRad = wp.theta * DEG_TO_RAD;
      const needleLen = 22;
      const needleX = pt.x + needleLen * Math.sin(headRad);
      const needleY = pt.y - needleLen * Math.cos(headRad);

      ctx.strokeStyle = isSelected ? '#60a5fa' : '#34d399';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(pt.x, pt.y);
      ctx.lineTo(needleX, needleY);
      ctx.stroke();

      // Needle tip handle
      ctx.fillStyle = isSelected ? '#3b82f6' : '#059669';
      ctx.beginPath();
      ctx.arc(needleX, needleY, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  drawPathTrail() {
    const trail = this.sim.pathHistory;
    if (trail.length < 2) return;

    const ctx = this.ctx;
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';

    for (let i = 1; i < trail.length; i++) {
      const p1 = this.toCanvas(trail[i - 1].x, trail[i - 1].y);
      const p2 = this.toCanvas(trail[i].x, trail[i].y);
      const speed = trail[i].v;
      const hue = clamp(180 - (speed * 110), 30, 200);
      ctx.strokeStyle = `hsla(${hue}, 85%, 55%, 0.75)`;

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
  }

  drawEKFEllipse() {
    const fused = this.sim.ekf.getPose();
    const pt = this.toCanvas(fused.x, fused.y);
    const ctx = this.ctx;

    const stdDevX = Math.sqrt(this.sim.ekf.P.get(0, 0)) * METER_TO_INCH * this.scale * 2.0;
    const stdDevY = Math.sqrt(this.sim.ekf.P.get(1, 1)) * METER_TO_INCH * this.scale * 2.0;

    ctx.strokeStyle = 'rgba(6, 182, 212, 0.45)';
    ctx.fillStyle = 'rgba(6, 182, 212, 0.08)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(pt.x, pt.y, Math.max(stdDevX, 4), Math.max(stdDevY, 4), 0, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }

  // Draw Authentic Holonomic X-Drive Robot with 45° angled omni wheels
  drawRobot() {
    const pt = this.toCanvas(this.sim.x, this.sim.y);
    const ctx = this.ctx;
    const thetaRad = this.sim.theta * DEG_TO_RAD;

    ctx.save();
    ctx.translate(pt.x, pt.y);
    ctx.rotate(thetaRad);

    const sizePx = 12.5 * this.scale;
    const half = sizePx / 2;
    const wheelW = 1.5 * this.scale;
    const wheelL = this.sim.wheelDiameterInches * this.scale * 0.9;

    // 1. Robot Base Plate
    ctx.fillStyle = '#1e293b';
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(-half, -half, sizePx, sizePx, 6);
    ctx.fill();
    ctx.stroke();

    // 2. Front Bumper Indicator (Cyan)
    ctx.fillStyle = '#06b6d4';
    ctx.fillRect(-half + 4, -half, sizePx - 8, 4);

    // 3. Intake Mechanism Animated Rollers
    ctx.fillStyle = this.sim.intakeVoltage !== 0 ? '#10b981' : '#334155';
    ctx.fillRect(-half * 0.6, -half - 3, sizePx * 0.6, 5);

    // 4. X-Drive Wheels: 4 Omni Wheels angled at 45 degrees
    const drawXWheel = (wx, wy, angleRad) => {
      ctx.save();
      ctx.translate(wx, wy);
      ctx.rotate(angleRad);

      ctx.fillStyle = '#0f172a';
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 1;
      ctx.fillRect(-wheelW / 2, -wheelL / 2, wheelW, wheelL);
      ctx.strokeRect(-wheelW / 2, -wheelL / 2, wheelW, wheelL);

      ctx.strokeStyle = '#94a3b8';
      for (let s = -wheelL / 2 + 2; s < wheelL / 2; s += 4) {
        ctx.beginPath();
        ctx.moveTo(-wheelW / 2, s);
        ctx.lineTo(wheelW / 2, s);
        ctx.stroke();
      }
      ctx.restore();
    };

    // Front-Left (45 deg) & Front-Right (-45 deg)
    drawXWheel(-half + 3, -half + 3, Math.PI / 4);
    drawXWheel(half - 3, -half + 3, -Math.PI / 4);

    // Back-Left (-45 deg) & Back-Right (45 deg)
    drawXWheel(-half + 3, half - 3, -Math.PI / 4);
    drawXWheel(half - 3, half - 3, Math.PI / 4);

    // 5. Heading Arrow
    ctx.strokeStyle = '#f43f5e';
    ctx.fillStyle = '#f43f5e';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, -half * 0.9);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, -half * 1.15);
    ctx.lineTo(-4, -half * 0.85);
    ctx.lineTo(4, -half * 0.85);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  }
}

// ============================================================================
// 8. Three.js 3D Viewport Engine & Onshape CAD Loader (GLTF, STL, OBJ)
// ============================================================================

// Lightweight IndexedDB storage for persistent CAD models
const CadStorage = {
  dbName: 'IRAlib_CAD_DB',
  storeName: 'cad_models',
  open() {
    return new Promise((resolve) => {
      if (!window.indexedDB) return resolve(null);
      const req = indexedDB.open(this.dbName, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(this.storeName);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    });
  },
  async save(key, data) {
    try {
      const db = await this.open();
      if (!db) return;
      const tx = db.transaction(this.storeName, 'readwrite');
      tx.objectStore(this.storeName).put(data, key);
    } catch (e) {
      console.warn("IndexedDB save failed:", e);
    }
  },
  async load(key) {
    try {
      const db = await this.open();
      if (!db) return null;
      return new Promise((resolve) => {
        const tx = db.transaction(this.storeName, 'readonly');
        const req = tx.objectStore(this.storeName).get(key);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => resolve(null);
      });
    } catch (e) {
      return null;
    }
  }
};

class ThreeFieldRenderer {
  constructor(container, simulator) {
    this.container = container;
    this.sim = simulator;
    this.isActive = false;
    this.cameraMode = 'iso'; // 'iso', 'top', 'follow'
    this.cadRotationOffset = 0;
    this.cadLoaded = false;
    this.modelMode = 'cad'; // 'cad' or 'procedural'

    if (typeof THREE === 'undefined') {
      console.warn("Three.js not loaded.");
      return;
    }

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x07090e);

    const rect = container.getBoundingClientRect();
    const aspect = (rect.width || 680) / (rect.height || 680);
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
    this.camera.up.set(0,0,1);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(rect.width || 680, rect.height || 680);
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);

    if (typeof THREE.OrbitControls !== 'undefined') {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
    }

    this.build3DField();
    this.build3DRobot();
    this.initCadLoaderUI();
    this.tryAutoLoadCad();
    this.setCameraPreset('iso');

    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    if (!this.renderer) return;
    const rect = this.container.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    this.camera.aspect = rect.width / rect.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(rect.width, rect.height);
  }

  build3DField() {
    // Ambient & Directional Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xffffff, 0.9);
    sunLight.position.set(40, -80, 100);
    sunLight.castShadow = true;
    this.scene.add(sunLight);

    const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.35);
    fillLight.position.set(-50, 60, 40);
    this.scene.add(fillLight);

    // 6x6 Field Tiles: 140.4" x 140.4" (Override)
    const fieldGeom = new THREE.PlaneGeometry(23.4, 23.4);
    const tileMaterials=[0x747a82,0x858b93].map(color=>new THREE.MeshStandardMaterial({color,roughness:.8}));
    for(let row=0;row<6;row++)for(let col=0;col<6;col++){
      const tile=new THREE.Mesh(fieldGeom,tileMaterials[(row+col)%2]);
      tile.position.set(-70.2+(col+.5)*23.4,-70.2+(row+.5)*23.4,0);
      tile.receiveShadow=true;this.scene.add(tile);
    }

    // Grid wireframe
    const grid = new THREE.GridHelper(140.4, 6, 0x384661, 0x222a3d);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.05;
    this.scene.add(grid);

    // Perimeter Wall (extruded polycarbonate look)
    const wallMat = new THREE.MeshStandardMaterial({
      color: 0x384661, transparent: true, opacity: 0.45, roughness: 0.2
    });
    const wallThick = 1.5;
    const wallHeight = 11.5;

    const makeWall = (w, h, x, y) => {
      const g = new THREE.BoxGeometry(w, h, wallHeight);
      const m = new THREE.Mesh(g, wallMat);
      m.position.set(x, y, wallHeight / 2);
      this.scene.add(m);
    };

    makeWall(140.4, wallThick, 0, 70.2);
    makeWall(140.4, wallThick, 0, -70.2);
    makeWall(wallThick, 140.4, -70.2, 0);
    makeWall(wallThick, 140.4, 70.2, 0);

    if(this.sim.override)OverrideView.build3D(this);
  }

  build3DRobot() {
    this.robot3D = new THREE.Group();

    // Container for imported Onshape CAD Model
    this.cadRobot = new THREE.Group();
    this.robot3D.add(this.cadRobot);

    // Container for Detailed Procedural VEX Override Robot Model
    this.proceduralRobot = new THREE.Group();
    this.buildDetailedProceduralRobot();
    this.robot3D.add(this.proceduralRobot);

    this.scene.add(this.robot3D);
  }

  buildDetailedProceduralRobot() {
    // Aluminum C-Channel Rails (15" x 14" chassis, brushed aluminum)
    const alumMat = new THREE.MeshStandardMaterial({ color: 0xc8d1dc, metalness: 0.85, roughness: 0.25 });
    const darkSteelMat = new THREE.MeshStandardMaterial({ color: 0x27272a, metalness: 0.7, roughness: 0.35 });
    const motorMat = new THREE.MeshStandardMaterial({ color: 0x18181b, roughness: 0.6 });
    const greenCartridgeMat = new THREE.MeshStandardMaterial({ color: 0x22c55e, emissive: 0x15803d, emissiveIntensity: 0.3 });
    const intakeRollerMat = new THREE.MeshStandardMaterial({ color: 0x06b6d4, roughness: 0.4 });
    const pneumaticMat = new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.9, roughness: 0.15 });

    // 1. Dual Main Longitudinal C-Channels
    const railGeom = new THREE.BoxGeometry(1.5, 14.0, 1.5);
    const leftRail = new THREE.Mesh(railGeom, alumMat);
    leftRail.position.set(-6.0, 0, 2.0);
    this.proceduralRobot.add(leftRail);

    const rightRail = new THREE.Mesh(railGeom, alumMat);
    rightRail.position.set(6.0, 0, 2.0);
    this.proceduralRobot.add(rightRail);

    // 2. Transverse Cross Rails
    const crossGeom = new THREE.BoxGeometry(10.5, 1.5, 1.5);
    const frontCross = new THREE.Mesh(crossGeom, alumMat);
    frontCross.position.set(0, 5.5, 2.0);
    this.proceduralRobot.add(frontCross);

    const backCross = new THREE.Mesh(crossGeom, alumMat);
    backCross.position.set(0, -5.5, 2.0);
    this.proceduralRobot.add(backCross);

    // 3. 4x V5 Smart Motors with Green Cartridges (45° angle mounts for X-Drive)
    const motorBoxGeom = new THREE.BoxGeometry(2.4, 1.5, 1.6);
    const cartGeom = new THREE.BoxGeometry(0.8, 1.2, 1.2);

    const addMotor = (x, y, angle) => {
      const mGroup = new THREE.Group();
      const mBody = new THREE.Mesh(motorBoxGeom, motorMat);
      const mCart = new THREE.Mesh(cartGeom, greenCartridgeMat);
      mCart.position.set(1.4, 0, 0);
      mGroup.add(mBody);
      mGroup.add(mCart);
      mGroup.position.set(x, y, 2.5);
      mGroup.rotation.z = angle;
      this.proceduralRobot.add(mGroup);
    };

    addMotor(-4.0, 4.0, Math.PI / 4);
    addMotor(4.0, 4.0, -Math.PI / 4);
    addMotor(-4.0, -4.0, -Math.PI / 4);
    addMotor(4.0, -4.0, Math.PI / 4);

    // 4. 4x Omni Wheels with Rollers at 45°
    const wheelGeom = new THREE.CylinderGeometry(2.0, 2.0, 1.4, 16);
    const wheelHubGeom = new THREE.CylinderGeometry(1.2, 1.2, 1.45, 12);
    const rollerMat = new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.5 });

    const add3DOmniWheel = (x, y, angle) => {
      const wGroup = new THREE.Group();
      const wRim = new THREE.Mesh(wheelGeom, rollerMat);
      wRim.rotation.y = Math.PI / 2;
      const wHub = new THREE.Mesh(wheelHubGeom, darkSteelMat);
      wHub.rotation.y = Math.PI / 2;
      wGroup.add(wRim);
      wGroup.add(wHub);
      wGroup.position.set(x, y, 2.0);
      wGroup.rotation.z = angle;
      this.proceduralRobot.add(wGroup);
    };

    add3DOmniWheel(-6.2, 6.0, Math.PI / 4);
    add3DOmniWheel(6.2, 6.0, -Math.PI / 4);
    add3DOmniWheel(-6.2, -6.0, -Math.PI / 4);
    add3DOmniWheel(6.2, -6.0, Math.PI / 4);

    // 5. Front Override Intake Mechanism (Uprights + Compliant Rollers)
    const towerGeom = new THREE.BoxGeometry(1.0, 1.0, 9.0);
    const leftTower = new THREE.Mesh(towerGeom, alumMat);
    leftTower.position.set(-3.5, 4.5, 5.5);
    this.proceduralRobot.add(leftTower);

    const rightTower = new THREE.Mesh(towerGeom, alumMat);
    rightTower.position.set(3.5, 4.5, 5.5);
    this.proceduralRobot.add(rightTower);

    // Intake Rollers (Upper & Lower)
    const rollerGeom = new THREE.CylinderGeometry(1.4, 1.4, 6.0, 16);
    const lowerRoller = new THREE.Mesh(rollerGeom, intakeRollerMat);
    lowerRoller.rotation.z = Math.PI / 2;
    lowerRoller.position.set(0, 5.0, 3.5);
    this.proceduralRobot.add(lowerRoller);

    const upperRoller = new THREE.Mesh(rollerGeom, intakeRollerMat);
    upperRoller.rotation.z = Math.PI / 2;
    upperRoller.position.set(0, 4.0, 8.5);
    this.proceduralRobot.add(upperRoller);

    // 6. Rear Override Object Mechanism (Brass cylinders + Steel hooks)
    const cylGeom = new THREE.CylinderGeometry(0.4, 0.4, 4.0, 12);
    const leftCyl = new THREE.Mesh(cylGeom, pneumaticMat);
    leftCyl.rotation.x = Math.PI / 2;
    leftCyl.position.set(-3.0, -6.5, 2.5);
    this.proceduralRobot.add(leftCyl);

    const rightCyl = new THREE.Mesh(cylGeom, pneumaticMat);
    rightCyl.rotation.x = Math.PI / 2;
    rightCyl.position.set(3.0, -6.5, 2.5);
    this.proceduralRobot.add(rightCyl);

    const clawGeom = new THREE.BoxGeometry(1.2, 3.5, 0.8);
    const leftClaw = new THREE.Mesh(clawGeom, darkSteelMat);
    leftClaw.position.set(-3.0, -8.5, 1.8);
    leftClaw.rotation.x = -0.3;
    this.proceduralRobot.add(leftClaw);

    const rightClaw = new THREE.Mesh(clawGeom, darkSteelMat);
    rightClaw.position.set(3.0, -8.5, 1.8);
    rightClaw.rotation.x = -0.3;
    this.proceduralRobot.add(rightClaw);

    // 7. V5 Robot Brain with LCD Display
    const brainBoxGeom = new THREE.BoxGeometry(4.0, 3.2, 1.2);
    const brainMat = new THREE.MeshStandardMaterial({ color: 0x09090b, roughness: 0.4 });
    const brain = new THREE.Mesh(brainBoxGeom, brainMat);
    brain.position.set(0, -1.0, 3.0);

    const screenGeom = new THREE.PlaneGeometry(3.2, 2.2);
    const screenMat = new THREE.MeshBasicMaterial({ color: 0x0284c7 });
    const screen = new THREE.Mesh(screenGeom, screenMat);
    screen.position.set(0, 0, 0.61);
    brain.add(screen);
    this.proceduralRobot.add(brain);

    // 8. V5 1100mAh Battery Pack
    const battGeom = new THREE.BoxGeometry(4.2, 1.8, 1.2);
    const battMat = new THREE.MeshStandardMaterial({ color: 0x27272a, roughness: 0.7 });
    const batt = new THREE.Mesh(battGeom, battMat);
    batt.position.set(0, -3.8, 1.5);
    this.proceduralRobot.add(batt);
  }

  initCadLoaderUI() {
    const fileInput = document.getElementById('cadFileInput');
    const btnToggleCad = document.getElementById('btnToggleCadModel');
    const btnRotateCad = document.getElementById('btnRotateCad');
    const dropOverlay = document.getElementById('cadDropOverlay');
    const wrapper = document.getElementById('canvasWrapper');

    if (fileInput) {
      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          this.loadCadFile(e.target.files[0]);
        }
      });
    }

    if (btnToggleCad) {
      btnToggleCad.addEventListener('click', () => {
        this.toggleModelMode();
      });
    }

    if (btnRotateCad) {
      btnRotateCad.addEventListener('click', () => {
        this.rotateCadModel();
      });
    }

    // Drag-and-Drop Handlers on Field Wrapper
    if (wrapper) {
      wrapper.addEventListener('dragenter', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (this.isActive && dropOverlay) dropOverlay.style.display = 'flex';
      });

      wrapper.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
      });

      wrapper.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.relatedTarget && !wrapper.contains(e.relatedTarget)) {
          if (dropOverlay) dropOverlay.style.display = 'none';
        }
      });

      wrapper.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (dropOverlay) dropOverlay.style.display = 'none';
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
          this.loadCadFile(e.dataTransfer.files[0]);
        }
      });
    }
  }

  async tryAutoLoadCad() {
    // 1. Check local models/robot.glb or models/robot.stl
    try {
      let res = await fetch('models/robot.glb');
      let ext = 'glb';
      let name = 'robot.glb';
      if (!res.ok) {
        res = await fetch('models/robot.stl');
        ext = 'stl';
        name = 'robot.stl';
      }
      if (res.ok) {
        const buffer = await res.arrayBuffer();
        this.loadCadFromBuffer(buffer, ext, name);
        return;
      }
    } catch (e) {}

    // 2. Check cached IndexedDB model from previous session
    try {
      const cached = await CadStorage.load('current_robot_cad');
      if (cached && cached.data) {
        this.loadCadFromBuffer(cached.data, cached.ext, cached.name);
        return;
      }
    } catch (e) {}

    // Default: use detailed procedural CAD robot
    this.cadRobot.visible = false;
    this.proceduralRobot.visible = true;
    const badge = document.getElementById('cadStatusBadge');
    if (badge) badge.textContent = "CAD: VEX V5RC Override";
  }

  loadCadFile(file) {
    const reader = new FileReader();
    const name = file.name;
    const ext = name.split('.').pop().toLowerCase();

    reader.onload = (e) => {
      const buffer = e.target.result;
      this.loadCadFromBuffer(buffer, ext, name);
      // Persist in IndexedDB for subsequent visits
      CadStorage.save('current_robot_cad', { name, data: buffer, ext });
    };

    if (ext === 'obj') {
      reader.readAsText(file);
    } else {
      reader.readAsArrayBuffer(file);
    }
  }

  loadCadFromBuffer(buffer, ext, filename) {
    const badge = document.getElementById('cadStatusBadge');
    if (badge) badge.textContent = `Загрузка ${ext.toUpperCase()}...`;

    const onModelReady = (model) => {
      this.applyCadModel(model, filename);
    };

    try {
      if (ext === 'glb' || ext === 'gltf') {
        if (typeof THREE.GLTFLoader === 'undefined') {
          console.error("GLTFLoader not found.");
          return;
        }
        const loader = new THREE.GLTFLoader();
        loader.parse(buffer, '', (gltf) => {
          onModelReady(gltf.scene || gltf.scenes[0]);
        }, (err) => console.error("GLTF Parse Error:", err));
      } else if (ext === 'stl') {
        if (typeof THREE.STLLoader === 'undefined') {
          console.error("STLLoader not found.");
          return;
        }
        const loader = new THREE.STLLoader();
        const geometry = loader.parse(buffer);
        const material = new THREE.MeshStandardMaterial({
          color: 0x94a3b8,
          metalness: 0.75,
          roughness: 0.35
        });
        const mesh = new THREE.Mesh(geometry, material);
        onModelReady(mesh);
      } else if (ext === 'obj') {
        if (typeof THREE.OBJLoader === 'undefined') {
          console.error("OBJLoader not found.");
          return;
        }
        const loader = new THREE.OBJLoader();
        const text = typeof buffer === 'string' ? buffer : new TextDecoder().decode(buffer);
        const obj = loader.parse(text);
        onModelReady(obj);
      }
    } catch (err) {
      console.error("CAD load error:", err);
      if (badge) badge.textContent = "Ошибка CAD";
    }
  }

  applyCadModel(modelObject, filename) {
    // Clear previous CAD children
    while (this.cadRobot.children.length > 0) {
      this.cadRobot.remove(this.cadRobot.children[0]);
    }

    // Enable shadows and enhance materials on all meshes
    modelObject.traverse((child) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
        if (!child.material) {
          child.material = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.6, roughness: 0.4 });
        }
      }
    });

    // Compute bounding box
    const initialBox = new THREE.Box3().setFromObject(modelObject);
    const initialSize = initialBox.getSize(new THREE.Vector3());

    // Scale to standard 18" VEX Override starting envelope
    const maxHorizontal = Math.max(initialSize.x, initialSize.y, initialSize.z);
    const targetSize = 17.5;
    const scale = (maxHorizontal > 0) ? (targetSize / maxHorizontal) : 1.0;
    modelObject.scale.set(scale, scale, scale);

    // Re-center horizontally and place on ground
    const scaledBox = new THREE.Box3().setFromObject(modelObject);
    const scaledCenter = scaledBox.getCenter(new THREE.Vector3());
    modelObject.position.x = -scaledCenter.x;
    modelObject.position.y = -scaledCenter.y;
    modelObject.position.z = -scaledBox.min.z + 0.1; // Ground clearance

    this.cadRobot.add(modelObject);
    this.cadLoaded = true;

    // Activate CAD mode
    this.modelMode = 'cad';
    this.cadRobot.visible = true;
    this.proceduralRobot.visible = false;

    const btnToggleCad = document.getElementById('btnToggleCadModel');
    if (btnToggleCad) btnToggleCad.textContent = "🤖 Режим: CAD (Onshape)";

    const badge = document.getElementById('cadStatusBadge');
    if (badge) {
      badge.textContent = `CAD: ${filename.substring(0, 16)}`;
      badge.style.borderColor = "#10b981";
      badge.style.color = "#34d399";
    }
  }

  rotateCadModel() {
    this.cadRotationOffset = (this.cadRotationOffset + Math.PI / 2) % (Math.PI * 2);
    this.cadRobot.rotation.z = this.cadRotationOffset;
    this.proceduralRobot.rotation.z = this.cadRotationOffset;
  }

  toggleModelMode() {
    const btnToggleCad = document.getElementById('btnToggleCadModel');
    if (this.modelMode === 'cad') {
      this.modelMode = 'procedural';
      this.cadRobot.visible = false;
      this.proceduralRobot.visible = true;
      if (btnToggleCad) btnToggleCad.textContent = "⚙️ Режим: Процедурный";
    } else {
      this.modelMode = 'cad';
      this.cadRobot.visible = true;
      this.proceduralRobot.visible = false;
      if (btnToggleCad) btnToggleCad.textContent = "🤖 Режим: CAD";
    }
  }

  setCameraPreset(mode) {
    this.cameraMode = mode;
    const btns = ['camIso', 'camTop', 'camFollow'];
    btns.forEach(id => {
      const b = document.getElementById(id);
      if (b) b.classList.remove('active');
    });

    if (mode === 'iso') {
      document.getElementById('camIso')?.classList.add('active');
      this.camera.position.set(115, -175, 200);
      this.camera.lookAt(0, 0, 0);
      if (this.controls) this.controls.target.set(0, 0, 0);
    } else if (mode === 'top') {
      document.getElementById('camTop')?.classList.add('active');
      this.camera.position.set(0, -0.01, 205);
      this.camera.lookAt(0, 0, 0);
      if (this.controls) this.controls.target.set(0, 0, 0);
    } else if (mode === 'follow') {
      document.getElementById('camFollow')?.classList.add('active');
    }
  }

  render() {
    if (!this.isActive || !this.renderer) return;

    // Update 3D robot transform
    if (this.robot3D) {
      this.robot3D.position.set(this.sim.x, this.sim.y, 0);
      this.robot3D.rotation.z = -this.sim.theta * DEG_TO_RAD; // 3D counter-clockwise
    }

    if(this.sim.override)OverrideView.update3D(this);
    // Update 3D mobile goals positions dynamically
    if (this.goalMeshes && this.sim.mobileGoals) {
      for (let i = 0; i < this.goalMeshes.length && i < this.sim.mobileGoals.length; i++) {
        const goal = this.sim.mobileGoals[i];
        this.goalMeshes[i].position.set(goal.x, goal.y, 0);
      }
    }

    // Follow camera mode
    if (this.cameraMode === 'follow') {
      const offsetDist = 45;
      const angle = -this.sim.theta * DEG_TO_RAD;
      const camX = this.sim.x - offsetDist * Math.sin(angle);
      const camY = this.sim.y - offsetDist * Math.cos(angle);
      this.camera.position.lerp(new THREE.Vector3(camX, camY, 30), 0.08);
      this.camera.lookAt(this.sim.x, this.sim.y, 4);
    } else if (this.controls) {
      this.controls.update();
    }

    this.renderer.render(this.scene, this.camera);
  }
}

// ============================================================================
// 9. Application Bootstrap, Jerry.io Planner & UI Binding
// ============================================================================
document.addEventListener('DOMContentLoaded', async () => {
  const canvas = document.getElementById('fieldCanvas');
  const threeContainer = document.getElementById('threeCanvasContainer');
  let core=null;
  try{core=await ProductionControl.load();}catch(error){console.error(error);}
  const sim = new VexRobotSimulator(core);
  window.nationalsSimulator=sim;
  document.getElementById('controlStatus').textContent=core?'C++ controller ready':'C++ unavailable — use HTTP server';
  const renderer = new FieldRenderer(canvas, sim);
  let threeRenderer = null;

  try {
    threeRenderer = new ThreeFieldRenderer(threeContainer, sim);
  } catch (err) {
    console.warn("3D initialization skipped:", err);
  }

  // 2D / 3D Mode Switcher
  const tab2D = document.getElementById('tab2D');
  const tab3D = document.getElementById('tab3D');
  const camera3DControls = document.getElementById('camera3DControls');

  tab2D.addEventListener('click', () => {
    tab2D.classList.add('active');
    tab3D.classList.remove('active');
    canvas.style.display = 'block';
    threeContainer.classList.remove('active');
    camera3DControls.style.display = 'none';
    if (threeRenderer) threeRenderer.isActive = false;
  });

  tab3D.addEventListener('click', () => {
    tab3D.classList.add('active');
    tab2D.classList.remove('active');
    canvas.style.display = 'none';
    threeContainer.classList.add('active');
    camera3DControls.style.display = 'flex';
    if (threeRenderer) {
      threeRenderer.isActive = true;
      threeRenderer.resize();
    }
  });

  // 3D Camera Controls
  document.getElementById('camIso')?.addEventListener('click', () => threeRenderer?.setCameraPreset('iso'));
  document.getElementById('camTop')?.addEventListener('click', () => threeRenderer?.setCameraPreset('top'));
  document.getElementById('camFollow')?.addEventListener('click', () => threeRenderer?.setCameraPreset('follow'));

  // Jerry.io Visual Waypoint Planner Controls
  const btnToggleJerry = document.getElementById('btnToggleJerry');
  const jerryBanner = document.getElementById('jerryBanner');
  const jerryPanel = document.getElementById('jerryPanel');
  const btnExitJerry = document.getElementById('btnExitJerry');
  const wpCountEl = document.getElementById('wpCount');
  const waypointTableBody = document.getElementById('waypointTableBody');
  const btnRunJerryPath = document.getElementById('btnRunJerryPath');
  const btnExportCpp = document.getElementById('btnExportCpp');
  const btnClearJerry = document.getElementById('btnClearJerry');

  let isJerryMode = false;

  const toggleJerryMode = (active) => {
    isJerryMode = active;
    if (isJerryMode) {
      // Force 2D view for editing
      tab2D.click();
      jerryBanner.style.display = 'flex';
      jerryPanel.classList.add('active');
      btnToggleJerry.classList.add('btn-primary');
      btnToggleJerry.classList.remove('btn-secondary');
    } else {
      jerryBanner.style.display = 'none';
      jerryPanel.classList.remove('active');
      btnToggleJerry.classList.remove('btn-primary');
      btnToggleJerry.classList.add('btn-secondary');
    }
  };

  btnToggleJerry.addEventListener('click', () => toggleJerryMode(!isJerryMode));
  btnExitJerry.addEventListener('click', () => toggleJerryMode(false));

  const syncWaypointTable = () => {
    const wps = renderer.jerryWaypoints;
    wpCountEl.textContent = wps.length;

    if (wps.length === 0) {
      waypointTableBody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align: center; color: var(--text-dim); padding: 1rem;">
            Кликните по полю, чтобы поставить первую точку!
          </td>
        </tr>`;
      return;
    }

    waypointTableBody.innerHTML = '';
    wps.forEach((wp, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="wp-num-badge">${idx + 1}</span></td>
        <td>${wp.x.toFixed(1)}"</td>
        <td>${wp.y.toFixed(1)}"</td>
        <td>${wp.theta.toFixed(0)}°</td>
        <td>
          <select data-idx="${idx}" class="wp-type-select" style="background: rgba(255,255,255,0.08); color: white; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; padding: 2px 4px; font-size: 0.75rem;">
            <option value="spline" ${wp.type === 'spline' ? 'selected' : ''}>Spline (Кривая)</option>
            <option value="strafe" ${wp.type === 'strafe' ? 'selected' : ''}>Strafe (Боком)</option>
            <option value="drive" ${wp.type === 'drive' ? 'selected' : ''}>Drive (Прямо)</option>
            <option value="turn" ${wp.type === 'turn' ? 'selected' : ''}>Turn (Разворот)</option>
          </select>
        </td>
        <td>${(wp.speed || 1.0).toFixed(1)}x</td>
        <td style="text-align: right;">
          <button data-idx="${idx}" class="wp-action-btn btn-del-wp" title="Удалить">✕</button>
        </td>
      `;
      waypointTableBody.appendChild(tr);
    });

    // Bind type dropdowns & delete buttons
    document.querySelectorAll('.wp-type-select').forEach(sel => {
      sel.addEventListener('change', (e) => {
        const i = parseInt(e.target.dataset.idx);
        wps[i].type = e.target.value;
      });
    });

    document.querySelectorAll('.btn-del-wp').forEach(b => {
      b.addEventListener('click', (e) => {
        const i = parseInt(e.target.dataset.idx);
        wps.splice(i, 1);
        syncWaypointTable();
      });
    });
  };

  // Canvas Mouse Interactions for Jerry.io Planner
  canvas.addEventListener('mousedown', (e) => {
    if (!isJerryMode) return;
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (canvas.height / rect.height);
    const fieldPt = renderer.toField(px, py);

    // Check click on existing waypoint
    let clickedWp = -1;
    let clickedHeading = false;

    for (let i = 0; i < renderer.jerryWaypoints.length; i++) {
      const wp = renderer.jerryWaypoints[i];
      const pt = renderer.toCanvas(wp.x, wp.y);
      const distPin = Math.hypot(px - pt.x, py - pt.y);

      // Check heading needle tip
      const headRad = wp.theta * DEG_TO_RAD;
      const needleX = pt.x + 22 * Math.sin(headRad);
      const needleY = pt.y - 22 * Math.cos(headRad);
      const distNeedle = Math.hypot(px - needleX, py - needleY);

      if (distNeedle < 10) {
        clickedWp = i;
        clickedHeading = true;
        break;
      } else if (distPin < 14) {
        clickedWp = i;
        break;
      }
    }

    if (clickedWp >= 0) {
      renderer.selectedWpIndex = clickedWp;
      renderer.draggedWpIndex = clickedWp;
      renderer.dragMode = clickedHeading ? 'heading' : 'pos';
    } else {
      // Add new waypoint
      const newWp = {
        id: renderer.jerryWaypoints.length + 1,
        x: clamp(fieldPt.x, -70, 70),
        y: clamp(fieldPt.y, -70, 70),
        theta: 0,
        type: renderer.jerryWaypoints.length === 0 ? 'drive' : 'spline',
        speed: 1.0
      };
      renderer.jerryWaypoints.push(newWp);
      renderer.selectedWpIndex = renderer.jerryWaypoints.length - 1;
      syncWaypointTable();
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (!isJerryMode || renderer.draggedWpIndex < 0) return;
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (canvas.height / rect.height);
    const wp = renderer.jerryWaypoints[renderer.draggedWpIndex];

    if (renderer.dragMode === 'pos') {
      const fieldPt = renderer.toField(px, py);
      wp.x = clamp(fieldPt.x, -70, 70);
      wp.y = clamp(fieldPt.y, -70, 70);
      syncWaypointTable();
    } else if (renderer.dragMode === 'heading') {
      const pt = renderer.toCanvas(wp.x, wp.y);
      const dx = px - pt.x;
      const dy = -(py - pt.y);
      wp.theta = Math.atan2(dx, dy) * RAD_TO_DEG;
      syncWaypointTable();
    }
  });

  window.addEventListener('mouseup', () => {
    renderer.draggedWpIndex = -1;
    renderer.dragMode = 'none';
  });

  btnClearJerry.addEventListener('click', () => {
    renderer.jerryWaypoints = [];
    renderer.selectedWpIndex = -1;
    syncWaypointTable();
  });

  // Run Custom Jerry.io Waypoint Route
  btnRunJerryPath.addEventListener('click', () => {
    const wps = renderer.jerryWaypoints;
    if (wps.length === 0) return;

    sim.resetSimulation();
    sim.isRunning = true;
    sim.setPose(wps[0].x, wps[0].y, wps[0].theta);

    for (let i = 1; i < wps.length; i++) {
      const prev = wps[i - 1];
      const cur = wps[i];

      if (cur.type === 'spline') {
        sim.queueAction({
          type: 'spline',
          start: { x: prev.x, y: prev.y, theta: prev.theta },
          end: { x: cur.x, y: cur.y, theta: cur.theta },
          maxVel: 1.0 * cur.speed, maxAccel: 1.8,
          desc: `WP ${i + 1}: Spline to (${cur.x.toFixed(0)}, ${cur.y.toFixed(0)})`
        });
      } else if (cur.type === 'strafe') {
        sim.queueAction({type:'pose',targetX:cur.x,targetY:cur.y,targetTheta:cur.theta,desc:`WP ${i+1}: strafe`});
      } else if (cur.type === 'turn') {
        sim.queueAction({
          type: 'turn',
          targetHeading: cur.theta,
          desc: `WP ${i + 1}: Snap Turn ${cur.theta.toFixed(0)}deg`
        });
      } else {
        sim.queueAction({
          type: 'drivePoint',
          targetX: cur.x, targetY: cur.y,
          desc: `WP ${i + 1}: Drive to (${cur.x.toFixed(0)}, ${cur.y.toFixed(0)})`
        });
      }
    }
  });

  // Export C++ Code Modal
  const exportModal = document.getElementById('exportModal');
  const cppCodePreview = document.getElementById('cppCodePreview');
  const btnModalClose = document.getElementById('btnModalClose');
  const btnModalDismiss = document.getElementById('btnModalDismiss');
  const btnCopyCode = document.getElementById('btnCopyCode');

  btnExportCpp.addEventListener('click', () => {
    const wps = renderer.jerryWaypoints;
    let code = `/**\n * @brief Autonomous Routine Generated by IRAlib Jerry.io Planner\n */\nvoid autoJerryCustomRoutine() {\n`;

    if (wps.length === 0) {
      code += `    // Нет путевых точек. Добавьте точки на поле!\n`;
    } else {
      code += `    chassis.setPose(${wps[0].x.toFixed(1)}, ${wps[0].y.toFixed(1)}, ${wps[0].theta.toFixed(1)});\n\n`;

      for (let i = 1; i < wps.length; i++) {
        const prev = wps[i - 1];
        const cur = wps[i];

        const x=cur.type==='turn'?prev.x:cur.x,y=cur.type==='turn'?prev.y:cur.y;
        const call=cur.type==='spline'?'autoSpline':'autoPose';
        code += `    if (${call}(${x.toFixed(1)}, ${y.toFixed(1)}, ${cur.theta.toFixed(1)}) != lemlib::MotionResult::Settled) return;\n`;

      }
    }
    code += `    controller.print(0, 0, "Custom Path Done!");\n}`;

    cppCodePreview.textContent = code;
    exportModal.classList.add('active');
  });

  const closeModal = () => exportModal.classList.remove('active');
  btnModalClose.addEventListener('click', closeModal);
  btnModalDismiss.addEventListener('click', closeModal);

  btnCopyCode.addEventListener('click', () => {
    navigator.clipboard.writeText(cppCodePreview.textContent);
    btnCopyCode.textContent = "Скопировано!";
    setTimeout(() => { btnCopyCode.textContent = "Скопировать в буфер"; }, 1500);
  });

  // UI Element References
  const routineSelect = document.getElementById('routineSelect');
  const btnPlay = document.getElementById('btnPlay');
  const btnPause = document.getElementById('btnPause');
  const btnReset = document.getElementById('btnReset');
  const btnOverrideStart = document.getElementById('btnOverrideStart');
  const btnOverrideReset = document.getElementById('btnOverrideReset');
  const btnStep = document.getElementById('btnStep');
  const speedButtons = document.querySelectorAll('.speed-opt');

  const brainLcdLines = [
    document.getElementById('brainLcd0'),
    document.getElementById('brainLcd1'),
    document.getElementById('brainLcd2'),
    document.getElementById('brainLcd3')
  ];

  const ctrlLcdLines = [
    document.getElementById('ctrlLcd0'),
    document.getElementById('ctrlLcd1'),
    document.getElementById('ctrlLcd2')
  ];

  const controllerShell = document.getElementById('controllerShell');

  const statX = document.getElementById('statX');
  const statY = document.getElementById('statY');
  const statTheta = document.getElementById('statTheta');
  const statVel = document.getElementById('statVel');
  const statLeftVolt = document.getElementById('statLeftVolt');
  const statRightVolt = document.getElementById('statRightVolt');

  const chartVel = document.getElementById('chartVel');
  const ctxVel = chartVel ? chartVel.getContext('2d') : null;

  btnPlay.addEventListener('click', () => {
    if (!sim.isRunning && sim.routineQueue.length === 0) {
      sim.startRoutine(routineSelect.value);
    }
    sim.isPaused = false;
  });

  btnPause.addEventListener('click', () => {
    sim.isPaused = !sim.isPaused;
  });

  btnReset.addEventListener('click', () => {
    sim.resetSimulation();
  });

  btnOverrideStart?.addEventListener('click', () => {
    if (sim.override) {
      sim.resetSimulation();sim.matchMode=true;
      const robot=sim.override.robots.find(r=>r.id===sim.activeRobotId);
      sim.setPose(robot.x,robot.y,robot.theta);
      const mode=document.getElementById('gameMode').value;
      sim.override.startMatch(mode);
      sim.controllerLcdLines=[mode==='practice'?'Override Practice':'Override Match',`Robot: ${robot.id}`,'Manual: driver phase'];
      document.getElementById('gameFeedback').textContent=mode==='practice'?'Тренировка без таймера. Движение и предметы доступны.':'Матч начат. Ручное управление доступно после автономной фазы.';
      sim.isPaused = false;
      sim.triggerRumble("..");
    }
  });

  btnOverrideReset?.addEventListener('click', () => {
    if (sim.override) {
      sim.resetSimulation();
      sim.triggerRumble(".");
    }
  });

  btnStep.addEventListener('click', () => {
    sim.isPaused = false;
    sim.update(0.01);
    sim.isPaused = true;
  });

  speedButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      speedButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      sim.timeScale = parseFloat(btn.dataset.speed || '0.5');
    });
  });

  document.getElementById('toggleTarget')?.addEventListener('change', (e) => { renderer.showTargetPath = e.target.checked; });
  document.getElementById('toggleTrail')?.addEventListener('change', (e) => { renderer.showTrail = e.target.checked; });
  document.getElementById('toggleEKF')?.addEventListener('change', (e) => { renderer.showEKF = e.target.checked; });
  document.getElementById('toggleCoord')?.addEventListener('change', (e) => { renderer.showCoordinates = e.target.checked; });

  const bindCtrlBtn = (id, callback) => {
    const btn = document.getElementById(id);
    if (!btn) return;
    btn.addEventListener('click', () => {
      btn.classList.add('pressed');
      setTimeout(() => btn.classList.remove('pressed'), 120);
      callback();
    });
  };

  bindCtrlBtn('btnCtrlA', () => {sim.isRunning=false;sim.routineQueue=[];sim.currentAction=null;sim.commandedWheelVoltages=null;sim.motorVolts=[0,0,0,0];sim.holonomicArcade(0,0,0);sim.lastMotionResult='Cancelled';});
  bindCtrlBtn('btnCtrlB', () => sim.startRoutine('testLinearDrive'));
  bindCtrlBtn('btnCtrlY', () => sim.startRoutine('testAngularTurn'));
  bindCtrlBtn('btnCtrlX', () => sim.startRoutine('testStrafe'));
  bindCtrlBtn('btnCtrlUP', () => sim.startRoutine('testQuinticSpline'));
  bindCtrlBtn('btnCtrlRIGHT', () => sim.startRoutine('testBezier'));
  bindCtrlBtn('btnCtrlDOWN', () => sim.startRoutine('testDiagonal'));
  bindCtrlBtn('btnCtrlL1', () => sim.startRoutine('autoHolonomicSkills'));

  document.getElementById('gameRobot')?.addEventListener('change',e=>{
    if(!sim.matchMode)return;
    sim.activeRobotId=e.target.value;const r=sim.override.robots.find(r=>r.id===sim.activeRobotId);sim.setPose(r.x,r.y,r.theta);sim.holonomicArcade(0,0,0);
  });
  document.querySelectorAll('[data-game-action]').forEach(button=>button.addEventListener('click',()=>{
    const result=!sim.matchMode?{ok:false,error:'Start Override first'}:sim.override.phase==='autonomous'?{ok:false,error:'Manual actions disabled during autonomous'}:sim.override.interact(sim.activeRobotId,button.dataset.gameAction,document.getElementById('gameObject').value);
    document.getElementById('gameFeedback').textContent=result.ok?'OK':result.error;
  }));
  // Keyboard Controls (Holonomic 3-DOF: W/S forward/backward, Q/E or A/D strafe, ArrowLeft/ArrowRight turn)
  const keysDown = {};
  window.addEventListener('keydown', (e) => {
    keysDown[e.key.toLowerCase()] = true;
    if (e.key === ' ') {
      sim.isPaused = !sim.isPaused;
      e.preventDefault();
    }
  });
  window.addEventListener('keyup', (e) => {
    delete keysDown[e.key.toLowerCase()];
  });

  // Animation Loop with delta accumulator for exact real-time speed & smooth playback
  let lastFrameTime = performance.now();
  let physicsAccumulator = 0;
  const PHYSICS_DT = 0.01; // 100 Hz fixed physics step

  function loop(currentTime) {
    if (!currentTime) currentTime = performance.now();
    let deltaSeconds = (currentTime - lastFrameTime) / 1000.0;
    lastFrameTime = currentTime;

    // Clamp to prevent lag jumps
    if (deltaSeconds > 0.1) deltaSeconds = 0.1;

    // Manual Holonomic 3-DOF Drive from Keyboard
    if (!sim.isRunning && !isJerryMode) {
      let forward = 0;
      let strafe = 0;
      let turn = 0;

      if (keysDown['w'] || keysDown['arrowup']) forward += 1.0;
      if (keysDown['s'] || keysDown['arrowdown']) forward -= 1.0;
      if (keysDown['d'] || keysDown['e']) strafe += 1.0;
      if (keysDown['a'] || keysDown['q']) strafe -= 1.0;
      if (keysDown['arrowright']) turn += 0.8;
      if (keysDown['arrowleft']) turn -= 0.8;

      sim.holonomicArcade(forward, strafe, turn);
    } else {
      sim.holonomicArcade(0, 0, 0);
    }

    // Accumulate time scaled by selected speed
    physicsAccumulator += deltaSeconds * sim.timeScale;

    // Step physics at fixed rate
    let substeps = 0;
    const MAX_SUBSTEPS = 15;
    while (physicsAccumulator >= PHYSICS_DT && substeps < MAX_SUBSTEPS) {
      sim.update(PHYSICS_DT);
      physicsAccumulator -= PHYSICS_DT;
      substeps++;
    }
    if (substeps >= MAX_SUBSTEPS) {
      physicsAccumulator = 0;
    }

    // Render active view (2D or 3D)
    if (!threeRenderer || !threeRenderer.isActive) {
      renderer.render();
    } else {
      threeRenderer.render();
    }

    // Update Override scoreboard and telemetry UI.
    if (sim.override) {
      const overrideState = sim.override.getState();
      const phase = document.getElementById("overridePhase");
      const clock = document.getElementById("overrideClock");
      const red = document.getElementById("overrideRed");
      const blue = document.getElementById("overrideBlue");
      if (phase) phase.textContent = overrideState.mode==='practice'&&!overrideState.matchEnded?'PRACTICE':overrideState.phase.replace("_", " ").toUpperCase();
      if (clock) {
        const remaining = Math.max(0, overrideState.rules.matchSeconds - overrideState.clock);
        clock.textContent = overrideState.mode==='practice'?'∞':Math.floor(remaining / 60) + ":" + String(Math.floor(remaining % 60)).padStart(2, "0");
      }
      if (red) red.textContent = String(overrideState.score.red);
      if (blue) blue.textContent = String(overrideState.score.blue);
    }    // Update Telemetry UI
    statX.textContent = `${sim.x.toFixed(1)}"`;
    statY.textContent = `${sim.y.toFixed(1)}"`;
    statTheta.textContent = `${sim.theta.toFixed(1)}°`;
    statVel.textContent = `${sim.v.toFixed(2)} m/s`;
    statLeftVolt.textContent = `${((sim.actionThrottle || 0) * 12).toFixed(1)} V`;
    statRightVolt.textContent = `${((sim.actionStrafe || 0) * 12).toFixed(1)} V`;

    for (let i = 0; i < 4; i++) {
      if (brainLcdLines[i]) brainLcdLines[i].textContent = sim.brainLcdLines[i];
    }

    for (let i = 0; i < 3; i++) {
      if (ctrlLcdLines[i]) ctrlLcdLines[i].textContent = sim.controllerLcdLines[i] || "";
    }

    if (sim.rumbleActive) {
      controllerShell?.classList.add('rumble-active');
    } else {
      controllerShell?.classList.remove('rumble-active');
    }

    if (ctxVel && sim.telemetry.time.length > 2) {
      const w = chartVel.width;
      const h = chartVel.height;
      ctxVel.clearRect(0, 0, w, h);

      ctxVel.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctxVel.beginPath();
      ctxVel.moveTo(0, h / 2);
      ctxVel.lineTo(w, h / 2);
      ctxVel.stroke();

      ctxVel.strokeStyle = '#06b6d4';
      ctxVel.lineWidth = 1.5;
      ctxVel.beginPath();
      for (let i = 0; i < sim.telemetry.vActual.length; i++) {
        const x = (i / 150) * w;
        const y = (h / 2) - (sim.telemetry.vActual[i] / 1.5) * (h / 2);
        if (i === 0) ctxVel.moveTo(x, y);
        else ctxVel.lineTo(x, y);
      }
      ctxVel.stroke();
    }

    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
});
