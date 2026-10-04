# Simulator and active control audit

Historical audit below predates the 2026-10-01 RobotAI extraction: original engine classes now live in `simulator/engine.js`, while `simulator/simulator.js` retains rendering/UI. The new shared original-Simulator benchmark and current parity/reset evidence are documented in [Motion episode contract](MOTION_EPISODE_CONTRACT.md); old file references/probes below describe the audited snapshot, not the current loader.

Audit date: 2026-09-28. Baseline HEAD: `f33c8392e163ac37814bb4e7cefdd5c229e89812`.
Scope: working-tree inspection in `Nationals-work3`, plus the bounded read-only probes below. Source facts, inferences and untested claims are distinguished. No robot was connected; no physical measurements are reported.

Only `docs/repository_map.json`, `docs/simulator_audit.md` and `docs/pedro_mapping.md` are owned by this audit. No applicable `AGENTS.md` was found in checked ancestors or recursive repository/reference searches. The user's deleted Pedro reference tree was not restored; the surviving sibling reference was read only. This is an audit handoff, not a claim that the RoboProof master plan is implemented.

## Key findings

1. **The active autonomous controller is the shared C++ cascade.** `src/main.cpp::autoPose` calls `nationals::followReference`; the XDRIVE branch in `src/subsystems/ltv/ltv.cpp::followPathImpl` converts spline states into that same boundary. The menu's Pedro Bezier routine constructs `BezierReference`; it does not execute `PedroFollower::follow`.
2. **The robot currently localizes using four drive motor encoders only.** `src/main.cpp` configures no IMU or external tracking sensors. `src/lemlib/chassis/odom.cpp` obtains XDRIVE position/heading from motor rotation deltas. A stationary chassis with spinning wheels can appear to move; a software settle result is not proof of physical arrival.
3. **WASM shares the controller/reference source, not the entire robot runtime.** `simulator/native/control.cpp` does not compile `main.cpp`, `odom.cpp`, `HolonomicMotion.cpp` or `safety.cpp`. The JS scheduler and synthetic feedback replace those boundaries.
4. **Repeatability has narrow positive evidence, realism does not.** Source/WASM hashes matched; two fresh fixed-step standalone runs produced identical selected traces. Plant parameter identification, full browser/match replay, ARM timing and physical path accuracy remain **UNVERIFIED**.
5. **Neither local follower is a Pedro Foresight port.** Geometry has shared mathematical primitives, but projection, time/braking laws, correction, allocation and completion differ. See `pedro_mapping.md`.

## Active robot path

| Stage | Source and observed behavior |
| --- | --- |
| Configuration | `config/robot.json` -> `tools/generate-config.js` -> `include/robot_config.hpp` and `simulator/robot-config.js`. Nominal 200 RPM, 3.25-inch wheels, 10.5-inch width/wheelbase, ratio 1; signed ports FL=-11, BL=-20, FR=1, BR=10. These are configuration values, not measured geometry. |
| Initialization | `src/main.cpp::initialize` configures `DriveOutput`, selects XDRIVE, calibrates/rebases odometry and starts monitoring/diagnostic/telemetry tasks. `src/lemlib/chassis/chassis.cpp::calibrate` creates motor-backed tracking-wheel wrappers when vertical sensors are null; these are not added physical sensors. |
| References | `Cascade.hpp::pointReference` uses a quintic time law with independent heading. `BezierReference.hpp` uses 512 arc-table intervals, tangent velocity and independent shortest-wrap heading, with sampled curvature slowdown. `QuinticSpline.cpp` uses a 2048-interval arc table and sampled tangential speed/acceleration/jerk feasibility checks. |
| Spline adapter | `src/subsystems/ltv/ltv.cpp` validates/interpolates trajectory samples and maps mathematical heading to clockwise north heading. XDRIVE delegates to `followReference`; the separate differential-drive Eigen LTV branch is not the selected main configuration. |
| Feedback boundary | `HolonomicMotion.cpp::readFeedback` reads an odometry snapshot and motor `get_actual_velocity_all()` values, converts inches/RPM into SI, and checks snapshot validity/epoch and finite wheel values. |
| Outer loop | `Cascade.hpp::Cascade` linearizes field kinematics about the reference, solves a discrete Riccati iteration with diagonal Q/R, caches gains, and produces body strafe/forward/yaw commands. This is not the legacy scalar class also called LQR elsewhere. |
| Inner loop | `WheelVelocity` applies X-drive inverse kinematics, uniform wheel desaturation, common acceleration ramp, `kS*tanh + kV*v + kA*a` feedforward and wheel PI. Filtered derivative-on-measurement is implemented, but `wheelKd=0` by default. Conditional integral update limits windup. |
| Output | `HolonomicMotion.cpp::followReference` sends four volts through `src/lemlib/safety.cpp::DriveOutput::wheelVoltages`, which converts to millivolts and calls PROS on signed ports. Exclusive tokens, mode checks, limits and failure-stop handling live here, outside WASM. |
| Termination | Target duration must elapse, then pose and velocity must remain within `Cascade.hpp::settled` thresholds for 150 ms. Timeout, cancellation, epoch/feedback faults and loss of output ownership terminate the motion; lease release stops/brakes motors. |
| Manual/diagnostic exception | `opcontrol` writes leased holonomic power commands directly; timed single-wheel diagnostics likewise bypass the cascade. The shared-cascade statement applies to the identified autonomous reference paths, not every motor command. |

Coordinate boundaries matter: public chassis poses are inches/degrees unless radians are requested; odometry snapshots are inches/radians; cascade signals use meters/radians/volts. Spline `State.heading` uses counterclockwise angle from +X, converted with `pi/2-heading`. Wheel order is always FL, BL, FR, BR in the active cascade/bridge.

### Localization and safety coverage

`src/lemlib/chassis/odom.cpp::readSample/update` uses four motor position deltas, an effective wheel diameter of physical diameter times sqrt(2), and turning span width+wheelbase. It integrates at midpoint heading and filters global/local velocity using a 25 ms EMA time constant. Stale samples over 100 ms and discontinuities latch faults; `setPose`/rebase changes the epoch. `getOdomSnapshot` carries timestamp, validity and epoch.

`lemlib::init` schedules odometry and `driveOutput().watchdog()` together at 10 ms. The watchdog rejects stale writes over 100 ms. **Source-derived risk:** a stalled odometry task also delays that watchdog; it is not an independent supervisor. Worst-case stop latency and PROS scheduling behavior are **UNVERIFIED**.

`src/subsystems/ekf/EKF.cpp`, `mcl/MCL.cpp`, `OdomReset.cpp` and `flc/FuzzyLogic.cpp` exist but are not instantiated/called as the active localization/control path in `main.cpp`. `MotorMonitor` is active monitoring, not pose fusion. Do not infer active EKF/MCL from file names, includes or simulator LCD labels.

## Simulator architecture and realism

`simulator/index.html` loads Override geometry/dynamics/autonomy/game/view scripts, then generated robot configuration, `control-runtime.js` and `simulator.js`. Three.js/rendering dependencies also come from remote CDNs. Standalone VM tests that load only `simulator.js` do not exercise this complete browser setup.

`simulator/simulator.js::executeAction` supplies the WASM controller with `odom`, `odomVelocity` and modeled wheel rim speeds. It does **not** supply plant truth x/y/theta to the motion controller. Truth remains available for rendering, contact resolution, game interactions and reports; this is an architectural separation, not an isolation/security boundary.

| Model element | Implemented source behavior | Fidelity limit |
| --- | --- | --- |
| Motor and supply | Per-wheel back-EMF/resistance/current clamp/torque; 48 V/s terminal-voltage slew; algebraic battery sag with a floor. | Constants are nominal hard-coded model parameters. Physical identification, actual V5 firmware limits and thermal behavior are **UNVERIFIED**. A current clamp is not a simulated thermal breaker. |
| Ground contact | Wheel slip stiffness, Coulomb traction cap, wheel inertia, implicit unsaturated wheel update and saturated traction branch. | Equal load per wheel, simplified roller resistance; no established load-transfer, compliance or calibrated tile variability. Stability comments are not a proof over all dt/parameters. |
| Chassis | 3-DOF planar Newton-Euler integration, rolling/yaw drag and low-speed unpowered lock. Clamped goal changes modeled mass/inertia/drag. | Not a full 3D robot or mechanism dynamics model. Mass 6.8 kg and yaw inertia 0.145 kg*m^2 are coded assumptions, not measurements from this audit. |
| Contact time step | `stepHolonomicPhysics` splits public steps larger than 0.0005 s into equal substeps. | At nominal 0.01 s this gives 20 substeps. The public direct plant method does not share all `update` input validation. Convergence across step sizes is **UNVERIFIED**. |
| Encoders | Rim speeds reconstructed from exact modeled wheel angular velocities feed kinematic odometry each contact substep; slip can separate odometry from truth. | No sampled encoder position packets, quantization, latency, dropout, finite stale values, PROS error codes or reset-epoch emulation on this path. |
| Velocity feedback | JS uses instantaneous synthetic odometry velocity and modeled wheel speed. | Hardware uses 10 ms position-delta odometry with EMA, plus a separate PROS RPM read. These are not identical sensing pipelines. |
| IMU/EKF | A JS `RobotEKF` predicts and updates from the synthetic odometry pose for display. | Controller uses `odom`, not `ekfPose`. `gyroDriftRate`/`gyroBias` are initialized/reset but do not inject IMU drift in the inspected plant. No independent IMU is configured on the robot. |
| Field geometry | `override-geometry.js::resolveRobot` uses rectangle/circle contact proxies and iterative penetration correction. Standalone fallback uses axis-aligned field bounds and velocity reflection. | Different environments between full match and standalone tests. CAD rendering is not mesh-based collision validation. |
| Game pieces | `override-dynamics.js` uses reduced vertical/tilt dynamics and deterministic contact/capture rules, with substeps capped at 0.005 s. | Simplified contact proxies, damping and capture thresholds; physical scoring fidelity and rules correctness are **UNVERIFIED** here. |
| Multi-robot | `override-autonomy.js::OverrideFleet` forks WASM instances, runs plants in map iteration order and advances shared game state. | Sequential updates are deterministic for a fixed setup but not a simultaneous coupled robot solver. Order-dependence was not tested. |

### Shared-source parity boundaries

- `tools/build-wasm.ps1` builds `simulator/native/control.cpp`, `QuinticSpline.cpp` and `BezierCurve.cpp`; included `Cascade.hpp` and `BezierReference.hpp` are shared production code. Ten tracked source hashes and a WASM hash are recorded in `control-build.json`.
- `ProductionControl.step` checks array lengths, then writes a constant valid flag of 1. C++ still rejects nonfinite data and invalid dt, but a finite stale sample cannot be represented as invalid via this JS API. There is no timestamp/epoch field in its feedback buffer.
- The browser's JS completion logic duplicates numerical thresholds from `settled` rather than calling firmware `followReference`. JS increments dwell by dt; firmware starts a clock when the first qualifying sample is observed. Boundary timing equivalence is **UNVERIFIED**.
- JS `configure` exposes radius/max wheel speed and computes kV; it does not expose every firmware cascade option. JSON linear/angular gains initialize legacy chassis settings, not all active cascade gains. JSON `drive`/`imuPort` do not automatically instantiate robot devices. Green motor gearing is explicit in `main.cpp`, while plant motor electrical constants are fixed separately.
- `initAction` clamps simulator spline requests to at most 0.45 m/s and 0.8 m/s^2. The firmware `autoSpline` accepts explicit supplied limits. Simulator point duration uses fixed defaults; firmware `autoPose` scales nominal speed by maxSpeed. Identical defaults do not prove parity for all API arguments.
- Legacy JS `LTVMath`, `QuinticSplineGenerator`, `VelocityController`, `LemLibLQRController` and `LemLibPIDController` remain in `simulator.js`; active motion execution uses `ProductionControl` instead. Testing a legacy helper is not coverage of the selected controller.

## Determinism: demonstrated versus unverified

**Source-confirmed design:** control inputs include explicit dt; the inspected WASM imports contain no clock/random source. No `Math.random` call was found in the active JS simulator/game files. Main browser physics uses a 0.01 s accumulator; fleet stepping subdivides to at most 0.01 s and splits at phase boundaries.

**Limits:** browser scheduling uses `performance.now`, clamps a rendered frame's elapsed time to 0.1 s and discards remaining accumulator time after 15 substeps. Keyboard events are sampled on render frames. Thus identical wall-clock input timing is not guaranteed to produce the same tick-indexed input stream. UI `setTimeout` callbacks control rumble/display/link cleanup. Cross-engine numeric equality, long-run reset equivalence, full fleet trace replay and asynchronous asset/UI behavior are **UNVERIFIED**.

The fleet report includes programs, game state, events and a bounded trace, not a demonstrated complete replay fixture with every input tick, sensor fault and build/config hash. Reproducibility requires those provenance fields and explicit initial conditions; a fixed step alone is insufficient.

## Checks actually performed

Runtime: **Node v24.19.0**. These checks made no repository writes.

| Check | Result | Exact scope |
| --- | --- | --- |
| `node tools/generate-config.js --check` | PASS | Existing generated header/JS match generator output. |
| SHA-256 manifest inspection | PASS | All 10 source files match after CRLF -> LF normalization; `control.wasm` byte hash matches `control-build.json`. This is freshness evidence, not an independent reproducible build. |
| `WebAssembly.Module.imports` | PASS | Observed only WASI `fd_close`, `fd_seek`, `fd_write`. |
| Fresh-instance repeat probe | PASS | VM loads `robot-config.js` and `simulator.js`, stubs DOM listener and timeout, omits Override. Each of two new simulators uses a separate WASM instance, starts at (0,0,0), queues `{type:'drive', targetInches:24, heading:0}`, sets isRunning and performs exactly 1000 `update(0.01)` calls. Both report `Settled`; JSON of the selected per-tick trace is exactly equal. |
| Local cubic formula probe | PASS | Points `[[0,0],[0,16],[24,8],[24,24]]` and `[[1,-2],[-9,14],[20,-30],[4,6]]`, each at t=0,0.1,0.25,0.5,0.9,1. Export `control_curve_eval` values and first/second derivatives agree with direct Bernstein formulas within 1e-4 per component. Java was not executed; curvature was not asserted in this probe. |

Repeat trace fields: simTime, truth x/y/theta, odometry x/y/theta, four motor voltages, four wheel angular velocities and lastMotionResult. Equality covers these fields only, in the same Node process. It is not a measured positional accuracy claim.

### Existing verification assets not run in full

- `tests/control-core.test.js`: manifest checks, closed-loop motion, slip, invalid-input/saturation/reset and cubic/spline checks. It writes `docs/recovery/control-results.json`, so it was **not run unchanged** under this three-file ownership restriction. Selected independent read-only checks above are not reported as a full-suite pass.
- `tests/safety.cpp`, `tests/motion.cpp`, `tools/test-safety.ps1`: exercise actual output/control boundary code against mocked PROS clock, motors and odometry. They do **not** compile real `src/lemlib/chassis/odom.cpp` into that harness. Execution this audit: **UNVERIFIED**.
- `simulator/test_physics.js`, `test_holonomic_drive.js`, `test_override_geometry.js`, `test_override_dynamics.js`, `test_override_match.js`: available plant/contact/match checks. Execution this audit: **UNVERIFIED**. Some physics checks print descriptive claims without asserting them; printed text is not an acceptance oracle. Its printed clamped-inertia example uses a different distance than the current plant formula.
- `tools/verify-arm-build.ps1`, firmware build and browser interactive testing: **UNVERIFIED**. Existing `docs/recovery/*` reports are historical artifacts, not newly reproduced evidence.

## Prioritized handoff risks

| Priority | Finding and impact | Required gate / suggested owner |
| --- | --- | --- |
| High | Encoder-only localization can settle in estimate space while physical pose is wrong under slip/contact. | Localization owner: independent pose/heading measurements or explicit accuracy limitations; bench slip/blocking trials with separate truth measurement. |
| High | Synthetic sensing and JS scheduling skip production odometry epochs, filtering, leases and watchdog. Passing a WASM motion scenario cannot establish fail-safe firmware behavior. | Simulator + verification owners: sensor packet/fault injection and actual odometry/boundary tests; PROS timing/stop-latency checks. |
| High | Motor, friction, inertia and battery parameters have no calibration evidence in this audit. | Plant owner: identify parameters from logged trials; report uncertainty and hold-out validation, not tuned-on-the-same-test accuracy. |
| Medium | Watchdog shares the odometry task. | Control owner: test stalled task/device calls and define independent stop authority if required. **UNVERIFIED** on hardware. |
| Medium | Configuration and API parity cover a subset of behavior, not all gains/gearing/arguments. | Integration owner: explicit parameter schema and parity cases for nondefault requests, gearing and custom cascade settings. |
| Medium | Full replay/input capture and contact-order independence are not demonstrated. | Simulator owner: tick-indexed commands/faults, initial state and config/build identity; repeated match and reset tests. |
| Medium | Spline feasibility is sampled and tangential; lateral acceleration/yaw/wheel constraints may saturate downstream. | Trajectory owner: chassis-feasibility checks and cusp/high-curvature regression cases. No failure frequency measured here. |
| Medium | Pedro labels and stale comments can imply algorithms not actually used. | Documentation/control owners: keep geometry reuse distinct from Foresight equivalence; see mapping. |

Do not restore the deleted reference or edit runtime files as part of this audit. Assign implementation/test changes to their respective owners, then update acceptance evidence after those changes are integrated.
