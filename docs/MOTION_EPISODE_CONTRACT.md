# Original-Simulator motion environment v2

Updated on `RobotAI`, 2026-10-02. Scripted baseline and bounded continuous policy actions now use the original Simulator and persistent production controller. This is a tested training environment, **not a trained movement policy**. Reports use schema version 2; old reports are historical evidence and must be regenerated for current-source replay.

## One authoritative engine

- `simulator/engine.js` contains the original math, estimator, robot physics and motion-execution classes extracted from `simulator/simulator.js`. The remaining `simulator.js` is the original rendering/planner/UI bootstrap. `simulator/index.html` loads the engine before the view.
- `simulator/motion.js` exposes the same `MotionEpisode` contract to browser scripts and CommonJS. `simulator/headless.js` creates isolated production WASM controllers and verifies their binary and C++ source hashes. `roboproof/motion.js` runs/replays that original engine, not `roboproof/sim.js`.
- Original Override geometry, dynamics, game and fleet modules remain shared and unchanged apart from propagating the host's robot configuration to fleet instances. No second plant, scoring implementation or Python physics exists in this path.
- Display data such as trails, LCD strings and bounded telemetry arrays remain derived engine data. Canvas/Three, DOM events, RAF and keyboard handling live in the view. Rumble lasts 0.3 **simulated** seconds, not a browser timer; pause therefore freezes it.
- The original plant is approximate. Software parity does not establish real-robot fidelity or validate the game's scoring rules.

## Audited update order and units

The outer simulation/controller tick is fixed at 0.01 seconds. The original engine internally subdivides motor/contact/rigid-body integration into steps no larger than 0.0005 seconds. Rendering may consume any number of completed ticks and does not determine an episode outcome.

At each outer tick, queued scripted motion is initialized if idle, its reference and encoder feedback are passed to the existing C++ cascade, wheel voltages are applied to the original motor/battery/contact plant, true motion and four-wheel odometry/EKF are integrated, then game synchronization (when enabled), clocks and display data are updated. Fleet mode retains its original four-controller dispatch and single shared game tick.

| Boundary | Convention |
| --- | --- |
| Original field pose and scripted targets | Inches; heading in degrees clockwise from +Y |
| Original physical velocity | Field-frame m/s; yaw rate rad/s clockwise |
| Controller point reference | `[x_m, y_m, heading_rad_CW_from_Y, vx_mps, vy_mps, yaw_radps]` |
| Controller feedback | Estimated pose 3 + estimated field velocity/yaw 3 + wheel rim speeds 4, all SI |
| Original spline conversion | Math heading from +X maps to `pi/2 - heading`; math yaw rate changes sign |
| Wheels | FL, BL, FR, BR |

`initAction()` resets the controller once per scripted motion or once when a policy-reference episode starts. Repeated policy decisions modify the reference data without replacing or resetting the C++ instance/integrators. Scripted mode continues to reject replacing a busy scripted command.

## Episode API

`new MotionEpisode(controlFactory)` requires a factory that returns a fresh `ProductionControl` instance. No DOM or timers are required.

- `reset(seed, configuration, task, options)` validates before construction, creates a fresh robot, game and WASM instance, resets clocks/queues/motors/odometry/EKF/display/metrics and policy adapter state, applies supported physical parameters and sets the initial pose. The seed is an unsigned 32-bit integer. Reset returns the first observation.
- `observe()` returns sensor-derived fields only, as described below. Reading observations does not consume randomness or advance the simulation.
- `applyAction({type: 'pose', xIn, yIn, headingDeg})` queues the existing absolute pose command. The robot is never teleported by an action. `{type: 'stop'}` requests neutral output; existing voltage slew/inertia still apply. At most 6,001 commands may be recorded per episode.
- `step(0.01)` advances exactly one original outer tick, returning `{observation, reward, rewardComponents, terminated, truncated, info}`. Any other dt is rejected and commands are neutralized without advancing time. Invalid action schemas/NaNs also neutralize commands and throw; an otherwise valid busy-command rejection leaves the active motion intact.
- `policyStep(values)` applies one bounded four-value action and advances up to the configured policy interval, summing actual rewards/components. It returns a 34-value sensor/reference vector and preserves `terminated` versus `truncated`, including a shorter final interval.
- `report()` records versioned options, seed/resolved configuration, engine/controller identities in the runner, action ticks, actual metrics and optional complete per-physics-tick transitions. Policy mode records transitions by default; scripted mode can request them. Reports are not trained-policy checkpoints.

## Frozen continuous-action interface

`simulator/policy.js` is shared between browser and headless execution. Select `controlMode: 'policy'`, with `policyIntervalTicks` 1–20 (default 5, i.e. 20 Hz over 100 Hz physics). `actionTimeoutTicks` defaults to twice that interval, at most 40 ticks. Missing or stale commands terminate unsuccessfully and neutralize output. Invalid policy actions are rejected, recorded as a safe stop, and terminate on the next tick. Commands must occur once on policy cadence boundaries. NaNs, infinities, sparse/wrong-sized arrays and values outside bounds are rejected, not silently clipped into valid commands.

Action is `{type: 'policy', values: [pace, right, forward, heading]}`, four finite numbers in [-1,1]. Pace maps to 0.25–1.0 of the existing production trajectory clock. The three residuals target bounded body-relative reference offsets: ±0.08 m in each translation axis and ±0.1 rad heading. A damped adapter limits residual speeds/accelerations, pace changes and reference field positions. The controller still executes the resulting six-value SI reference with sensor feedback; actions never teleport the robot or directly bypass the controller to command motors.

The nominal fixture is `[1,0,0,0]`. It is a scripted interface test, not a neural agent. This first action space adapts pace and residual references around an existing point-to-point profile; it is not a general waypoint/path planner, obstacle avoider or a controller-switching system. A later action-contract expansion requires separate tests and evaluation, not silent changes during learning.

Policy identity is declared as `{kind: 'untrained'|'scripted'|'learned', id, sha256?}`. Learned declarations require a checkpoint hash, but declarations alone do not prove weights were loaded or trained; the future trainer must retain that evidence independently.

Policy observations add known adapter reference, clock/pace, offsets/offset velocities and action age. `NationalsPolicy.vector()` fixes the 34-value order: pose 3, sensor velocity 3, wheels 4, target error 4, previous policy action 4, remaining time 1, reference 6, reference clock 1, pace 1, offsets 3, offset velocities 3, action age 1. No evaluator truth is included. Invalid sensor observations at a fault terminal return the last valid observation, with `info.observationValid: false`; this is an explicit unsuccessful fallback, not a valid fresh measurement.

Each retained transition has pre-action observation, applied action, actual reward/components, next observation, termination/truncation, evaluator-only metrics and physical pose/velocity/voltages for replay. Decision-boundary observations match those actually seen by the policy, not an observation rewritten after accepting its action. The future trainer should group physics ticks into policy decisions and use `policyStep` rewards/durations consistently.

## Python bridge and integrated interface

`roboproof/motion-bridge.js` is a bounded local JSON-lines process. `roboproof/motion_env.py` owns it with shell-free argv, timeouts, response identity checks and explicit close. Python performs no physics. Standard-library `MotionEnvironment.reset()` returns `(vector, info)` and `step(action)` returns `(vector, reward, terminated, truncated, info)`. This API shape does not claim it is already a registered Gymnasium environment or a trained RL implementation.

The dashboard's Motion lab runs, saves and restores original-engine experiments; fixtures are labeled untrained. Exact server replay recomputes recorded actions without calling the policy or any model. `/simulator/index.html?motion=<saved-id>` loads the same world/controller into the original 2D/3D view, checks browser-source/WASM hashes and compares every recomputed transition with 1e-9 absolute tolerance. It does not replace physics with an animation. Play/pause/reset/step/speed remain available; unrelated robot controls are disabled during replay. Browser tolerance checks are separate from exact same-runtime Node replay.

The task is `{start: {xIn, yIn, headingDeg}, goal: {xIn, yIn, headingDeg}, deadlineSeconds}`. Both positions are limited to [-60, 60] inches and headings to [-180, 180] degrees. Deadline is a multiple of 0.01 seconds between 0.01 and 60 seconds. Default: (0,0,0) -> (0,24,0), 10 seconds.

This reach task preserves the original game's inventory but runs outside match mode. Original perimeter contact is active; game-object obstacles, four-robot interactions, lift/pickup/placement and scoring are **not part of this reward/task**. Those require the later game-task gate rather than silently changing this benchmark.

## Supported physical variation

Each configuration value may be a fixed number or `[minimum, maximum]`. Ranges are sampled once at reset, in a fixed parameter order, using an episode-local LCG (`1664525 * state + 1013904223` modulo 2^32). It does not touch `Math.random`. Resolved values are retained for replay.

| Existing plant property | Accepted experiment bounds |
| --- | --- |
| `massKg` | 3.4–13.6 |
| `moiKgM2` | 0.0725–0.58 |
| `muLong` | 0.3–1.1 |
| `muLat` | 0.02–0.15 |
| `kSlip` | 325–975 |
| `batteryInternalR` | 0–0.12 |

These are bounded software experiments, not calibrated physical probability distributions. Unspecified properties retain original values. Geometry/gearing, sensor noise, latency and faults from the old reduced-model schema are not accepted here. `batteryVoltage` is deliberately rejected: the original plant recomputes it from current and internal resistance each substep. Its inactive `gyroDriftRate` field is not advertised as functioning noise/randomization.

## Observations versus evaluator truth

Observations contain:

- `pose`: encoder-odometry x/y in meters and clockwise heading in radians.
- `velocity`: encoder-derived field x/y m/s and yaw rad/s.
- `wheelRimSpeeds`: four measured wheel speeds in m/s.
- `targetError`: target-relative body-right/body-forward error in meters, then sine/cosine of wrapped heading error; computed with the estimated pose, never true pose.
- `previousAction`: the previous validated command or null; `remainingSeconds`: tick-based time budget.

`info`/`metrics` are a **separate evaluator channel** with true endpoint/heading error, physical speed and yaw. They must not be fed to a sensor-realistic policy. The simulated wheel/encoder path is still approximate, not a claim of realistic noisy sensors.

## Success, truncation and measured reward

Success requires true position error <0.02032 m, wrapped true heading error <0.035 rad, true linear speed <0.0254 m/s and true yaw rate <0.0873 rad/s for 15 consecutive outer ticks (0.15 s). The C++ controller's `Settled` label alone is not success. A deadline without success is `truncated: true`, not a successful terminal. Nonfinite state or existing controller/sensor faults are unsuccessful terminals. No stepping after either kind of episode end is accepted.

Per-tick reward components:

- Progress: previous minus current true goal distance in meters.
- Time: `-0.01 * dt`.
- Effort: `-0.0001 * sum(abs(motorVolts * motorCurrents)) * dt`.
- Contact: `-0.1 * delta(contactSeconds)`, counted from actual original contact normals over physical substeps.
- Success: +1 on the success transition only; fault: -1 on a fault terminal.

The voltage/current measure is explicitly an **effort proxy**, not an exact battery-energy integral. Contact seconds are not a count of unique collisions. Path length is measured from outer-tick true positions. These definitions are fixed before learning; a later policy cannot redefine success by moving a marker or trusting a predictor.

## Baseline and independent evaluation

Baseline: one existing pose command executed by the production LTV-LQR -> wheel PI/feedforward cascade. This is not an AI and not proof of globally optimal motion.

`tests/fixtures/motion-evaluation-worlds.json` freezes 12 evaluation worlds: four cardinal 24-inch moves, a diagonal/90-degree goal and a heading-wrap move, each nominal and with seeded mass/friction/battery-resistance variation. Do not use them for training, curriculum tuning or hyperparameter selection. Future policy comparisons must use identical worlds/deadlines/controller limits, frozen policies and independent training seeds.

`motion-evaluation-worlds-v2.json` freezes 24 additional worlds in eight task groups: varied starts/headings, heading-only motion, longer travel, deadline pressure and supported load/traction/power changes. It is deliberately harder. Some deadline-pressure cases can exceed the nominal profile's available time; failures are retained, not reclassified as learning wins. No obstacles or game score are implied. `assertTrainingWorld()` excludes frozen IDs, seeds and task definitions from future training/tuning. The current evaluator compares only the controller baseline and nominal scripted adapter; it does not execute learned weights.

The predeclared v2 acceptance contract requires zero success regressions, no contact increase, at least 5% mean time or effort improvement across all worlds for efficiency claims, and three independent training seeds before learned-improvement evidence. All-world means include failures/timeouts, rather than cherry-picking successful cases. Future learning reports must implement and enforce this gate rather than trusting a larger training reward.

Predeclared metrics are success rate (primary), endpoint/heading error, elapsed simulated time, contact seconds, effort proxy and path length. Any improvement claim must show baseline and learned-policy distributions, with no unacceptable success/safety regression; lower time on selected successes alone is insufficient. Larger obstacle/game evaluation needs additional fixtures and validated sensing/rules.

From the repository root:

```powershell
node roboproof/cli.js motion-baseline --seed 42 --out roboproof/runs/motion/baseline.json
node roboproof/cli.js motion-replay --file roboproof/runs/motion/baseline.json --out roboproof/runs/motion/replay.json
node roboproof/cli.js motion-check
node roboproof/cli.js motion-evaluate --suite 2
node --test tests/simulator-engine.test.js
& tools/start-simulator.ps1 -Port 8765
```

Optional `--task task.json` and `--configuration configuration.json` accept only this contract, not a legacy RoboProof scenario. Replay checks engine/game/config/runtime/WASM identities and exact recorded results in the same runtime. Cross-runtime equality and arbitrary hidden-controller snapshots are not promised. CLI reports are under ignored `roboproof/runs`; no robot firmware, paid service or cloud API is modified/called.

## Historical benchmark evidence and current gate

- Captured pre-extraction original traces for forward, strafe, heading-wrap pose, wall obstruction and a four-robot match. All 5 full trace digests match after extraction; their match score remains 11:11. Signed zero is retained in live parity comparisons and normalized only by the existing JSON fixture serialization.
- 13 new tests pass: DOM-free import, unchanged traces, exact browser-realm/headless episode transitions, interleaved isolated controllers and seeded resets, sensor/truth separation, success/truncation/reward semantics, invalid inputs, simulated rumble and identity-checked replay.
- Scripted evaluation: 12/12 frozen worlds succeed. Default 24-inch target reaches measured success at 2.91 simulated seconds with 0.00121016 m endpoint error. These are baseline results, **not learning improvement**.
- Full existing/new Node regression run: 75 pass, 0 fail, 1 external Java-reference test explicitly skipped because the available Java 8 installation lacks a usable compiler. Earlier retained Java differential evidence is separate; this run does not claim to repeat it.
- On 2026-10-02 the v2 readiness smoke passed: nominal scripted baseline and policy fixture succeed; bounded random actions truncate correctly; JSON-round-trip replay matches; stale and invalid actions stop safely. Additional tests cover persistent state, limits, browser/headless policy parity, Python ownership, stored API replay, UI no-automatic-execution and altered-trace rejection.
- Fresh v2 frozen evaluation: controller baseline 6/24 success, nominal adapter 7/24. This exposes limited robustness in the harder software worlds. Neither agent learns; the difference is not learned improvement and must not be marketed as such.
- Next: select a compact CPU motion learner, implement bounded gradient/checkpoint/resume execution, then evaluate frozen policies on independent worlds/seeds. Game-score training and physical/cloud deployment remain gated.
