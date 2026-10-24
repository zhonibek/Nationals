# 🛠️ IRAlib complete robot tuning guide (step-by-step calibration tutorial)

This guide is a complete step-by-step description of calibrating and tuning every **IRAlib** control system (track width, feedforward $kS/kV$, LQR, PID, FLC, TCS, LTV and EKF) on a VEX V5 competition robot.

---

## 📋 Checklist before calibration

1. **Battery charge:** at least 85% (voltage > 12.4 V).
2. **Surface:** clean competition foam tiles.
3. **Mechanics:** check that the wheels spin freely and that chains/gears are properly tensioned.
4. **Constants in `src/main.cpp`:**
   - Motor cartridges: `pros::MotorGearset::blue` (600 RPM) or `pros::MotorGearset::green` (200 RPM).
   - `DRIVETRAIN_RPM`: the actual wheel speed including gearing (for example, $600 \times \frac{36}{48} = 450.0\,\text{RPM}$).
   - `WHEEL_DIAMETER_INCHES`: wheel diameter (for example, `3.25` for new omni wheels).

---

## 🎮 Interactive calibration buttons on the controller

In **Driver Control (`opcontrol`)** mode the controller has hotkeys for every test:

```
                  ┌──────────────────────┐
                  │      V5 CONTROLLER   │
                  │                      │
   [ UP ] LTV S-Curve    │                      │  [ X ] Mode (LQR <-> PID)
 [LEFT]             │   ( LCD SCREEN )     │ [ Y ] 90° Turn Snap
 [RIGHT] Quintic Spline  │                      │  [ A ] 360° Track Width
  [DOWN] kS Friction     │                      │  [ B ] 24" Linear Drive
                  └──────────────────────┘
```

---

## 📐 STEP 1. Track width calibration — button [ A ]

Track width (`TRACK_WIDTH_INCHES`) determines odometry accuracy and turn consistency for a differential drive.

1. Place the robot on a line intersection of the competition field.
2. Press **[ A ]**. The robot makes one full 360° turn in place.
3. **Analyse the result:**
   * If the robot **under-rotates** short of the starting line (for example, turns 345°):
     👉 **DECREASE** `TRACK_WIDTH_INCHES` in `src/main.cpp` (for example, from `10.5` to `10.2`).
   * If the robot **over-rotates** past the line (for example, turns 375°):
     👉 **INCREASE** `TRACK_WIDTH_INCHES` in `src/main.cpp` (for example, from `10.5` to `10.8`).
4. Repeat until the robot stops within $1^\circ$.

---

## ⚡ STEP 2. Static friction calibration ($k_S$) — button [ DOWN ]

$k_S$ is the minimum voltage, in volts, needed to overcome the static friction of the gearboxes and tyres.

1. Place the robot on a flat part of the field.
2. Press the **[ DOWN ]** arrow.
3. The robot slowly ramps up motor voltage and detects the breakaway threshold.
4. The controller screen shows the value: `kS = 0.45 Volts`.
5. Enter the measured number in `velConfig.KS_straight` in `src/main.cpp`:
   ```cpp
   velConfig.KS_straight = 0.45; // your measured value
   velConfig.KS_turn = 0.35;     // usually about 20% lower than straight
   ```

---

## 📏 STEP 3. Tuning the linear LQR controller (24 inches) — button [ B ]

Responsible for fast and accurate acceleration and braking in a straight line (`moveToPoint`).

1. Place the robot along a field line and press **[ B ]**.
2. The robot drives 24 inches (one tile). The controller shows: `Err: +0.15in 580ms`.
3. **Tuning rules (`lateralLQR` in `src/main.cpp`):**
   * **$k_P$ (stiffness / proportional gain):**
     * If the robot stops short of the target (under-travel) $\rightarrow$ **increase $k_P$** (from 10.0 to 12.0–14.0).
     * If the robot vibrates or jerks $\rightarrow$ **decrease $k_P$**.
   * **$k_V$ (velocity damping):**
     * If the robot overshoots the target and sharply backs up $\rightarrow$ **increase $k_V$** (from 3.2 to 4.0–4.5).
     * If the robot brakes too early and creeps over the last centimetres $\rightarrow$ **decrease $k_V$**.

---

## 🔄 STEP 4. Tuning the angular LQR controller (90° turn) — button [ Y ]

Responsible for a fast, clean turn to a target heading (`turnToHeading`).

1. Press **[ Y ]**. The robot makes a fast 90° turn.
2. The controller shows: `Err: 0.4deg 420ms`.
3. **Tuning rules (`angularLQR` in `src/main.cpp`):**
   * **$k_P$ (speed of reaching the heading):**
     * If the turn is slow $\rightarrow$ increase $k_P$ (from 1.0 to 1.5–2.0).
   * **$k_V$ (damping of rotational inertia):**
     * If the robot oscillates at the end of the turn ("fishtailing") $\rightarrow$ **increase $k_V$** (from 1.8 to 2.4–2.8).
     * If the robot vibrates during the turn $\rightarrow$ **decrease $k_V$**.

---

## 🏎️ STEP 5. Testing traction control (TCS) and S-curves

### 1. LTV-LQR trajectory — button [ UP ]
* Press the **[ UP ]** arrow. The robot follows a continuous 2-metre S-curve, solving the discrete algebraic Riccati equation (DARE) every 10 ms.

### 2. Jerk-limited fifth-order quintic spline — button [ RIGHT ]
* Press the **[ RIGHT ]** arrow. The robot builds a smooth fifth-order spline ($C^2$ continuity with jerk $j \le 3.5\,\text{m/s}^3$) and follows it without jerks.

### 3. Comparing LQR with classic PID — button [ X ]
* Press **[ X ]** to switch the chassis to classic PID, then repeat tests **[ B ]** and **[ Y ]** to compare how smoothly the robot stops.

---

## 📊 Telemetry on the V5 Brain LCD

While running, the V5 Brain screen shows:
* `EKF X, Y, Theta` — coordinates filtered by a 5-state extended Kalman filter.
* `Odom X, Y` — raw encoder coordinates.
* `TCS: ACTIVE` — traction control system indicator.
* `FLC: ON` — adaptive fuzzy-logic control indicator.
