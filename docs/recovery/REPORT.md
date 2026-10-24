# Nationals-work3 recovery — 19 September 2026

## Result and scope

The X-drive competition flow uses a persistent cascade:

`point / quintic spline / Pedro Bézier → LTV-LQR → four wheel-velocity targets → PID + kS/kV/kA → motor voltage`.

LTV-LQR here is a single outer controller: the model is linearized along the reference motion and the LQR gains are computed through DARE. It is not switching between separate LTV, LQR and PID controllers. The four inner controllers run on every update of the outer loop. By default D=0: PI + feedforward, as in the supplied 1516X example. The transition to holding the final pose keeps the same cascade.

**Work on a full copy of Override is not finished.** The result is an executable test bench with the production C++ core, a motion model and a corrected game layer. Exact CAD geometry, full stack dynamics, the robot's mechanisms and the whole V5 firmware are not reproduced in the browser. No tests on the physical robot were performed.

## What was compared

- The folder `scratch/Nationals-work3` was fixed, starting from version `e93922f`.
- `scratch/Nationals` and `scratch/successful models` were used as read-only sources.
- The remote original Nationals version `a9113ca` on `origin/main` was checked. The histories of that repository and the current local work3 copy have no common ancestor; therefore the result should be kept on a separate branch without force-replacing `main`.
- In the original Nationals, the spline already used `LTVPathFollower → VelocityController`. Other helpers selected separate controllers. So claiming that the whole original was always a persistent cascade is incorrect. The new architecture meets the user's stricter, clarified requirement.
- The two-wheel velocity loop in `successful models/LTV-LQR + PID/push_back_x-main` was studied. An X-drive needs a three-input motion model and four wheel controllers; a two-input tank-drive controller cannot command independent strafing without changing the model.
- The geometry of the Java Pedro Pathing library and the C++ Bézier implementation was compared. The C++ numbers were checked against an independent Bernstein formula. Running the Java code and a full differential test of the whole Pedro follower were **not performed**.

Pre-existing deletions of `PedroPathing-main/**` are not part of this fix. The original directories were not rewritten.

## Defects and changes

| Severity | Problem | Fix and main files |
|---|---|---|
| High | The working `main.cpp` did not include the LTV cascade in the competition flow | Added an `LTVPathFollower` instance, the autonomous sequence, cancellation between steps and LCD route selection; `src/main.cpp`, `include/autonomous.hpp` |
| High | LTV rejected the X-drive; the two-input tank model cannot strafe independently | A separate three-input X-drive LTV model and four inner loops; `Cascade.hpp`, `HolonomicMotion.cpp`, `ltv.cpp` |
| High | Point/turn/Pedro could bypass the cascade | Helpers and Pedro geometry feed one C++ loop. The preliminary `turnFirst` for the X-drive also uses this loop; `main.cpp`, `ltv.cpp`, `BezierReference.hpp` |
| High | The simulator tested its own outdated JS version of the controller | The production C++ is compiled to `control.wasm`; JS passes measurements and receives volts. Source hashes are checked by a test; `simulator/native/control.cpp`, `control-runtime.js`, `tools/build-wasm.ps1` |
| High | Odometry coordinates depended on the model's true motion | Odometry receives wheel velocities, integrates its own estimate and can drift during slip; `simulator/simulator.js` |
| High | The model could stay still under non-zero commands to four motors | The rest hold checks the actual wheel targets. Double counting of chassis inertia in the rotor dynamics was removed; traction limits rotor acceleration; `simulator.js` |
| High | Voltage went through the power-command API | Added `voltage` and `wheelVoltages` with checks and a single output owner; LTV writes volts directly; `safety.hpp`, `safety.cpp`, `ltv.cpp` |
| High | Running out of time could look like a successful finish | Success requires a small error and speed for 150 ms. Timeout/error stops the queue; `HolonomicMotion.cpp`, `simulator.js` |
| Medium | A PROS task creation failure was checked only through the C++ object pointer | The internal task handle is checked; `Busy` is returned and waiting does not stay `Running`; `ltv.cpp` |
| Medium | Travelled distance was computed as displacement from the start point | Lengths of consecutive segments are accumulated; `ltv.cpp` |
| Medium | The planner export referenced missing helpers | Calls to the existing `autoPose` / `autoSpline` with result checks are generated; `simulator.js` |
| Medium | The simulator increased an explicit timeout and rotated a relative diagonal by the final heading | Explicit deadlines are respected; demonstrations get a profile-based deadline. Relative translation uses the initial orientation, the final heading is independent; `simulator.js`, `control-core.test.js` |
| Medium | **Step** did not advance physics while paused | One tick runs and the simulation returns to pause; `simulator.js` |
| High for scoring | The autonomous bonus was recalculated after the driver period | Snapshot at the 15-second boundary; the final score is frozen separately; `override.js`, `test_override.js` |
| Medium for scoring | Cups were modelled as separate opaque/transparent objects; field contents and visibility were wrong | 63 Pins, 56 Cups, both halves on every Cup, starting stacks and placement per FO-1/FO-2; checks of alternation and visible halves; `override.js` |
| Medium for scoring | Midfield contact was checked as a square; wall contact was a flag | SAT for a rotated rectangle and a diamond; wall contact is computed from geometry. Starting robots really touch the wall; `override.js` |
| Medium | 3D showed the previous game elements | 2D and 3D render one Override state; added Pins, Cups, nine goals, Toggles, loaders, four robots and action buttons; `override-view.js`, `index.html` |

Drive ownership protection, the watchdog, the competition-mode check and cancellation existed in work3 before this change. They were kept, extended for voltage and covered by tests; they should not be attributed entirely to the new work.

## Checks and measurements

The commands to reproduce are listed in the root README. Full numeric results: [control-results.json](control-results.json).

| Check | Result |
|---|---|
| `node tests/control-core.test.js` | 17 scenarios: strafing both ways at 0/90/180/270°, forward/backward, combined turn and translation, 359→1° wrap, spline, Bézier, increased mass and reduced traction |
| C++ core edge cases | Invalid dt, NaN, saturation, reset, wheel speed limits, explicit short/invalid timeout, WASM freshness |
| Geometry | C++ quintic and Bézier; P/P′/P″ compared with an independent formula on two curves at six points |
| `tools/test-safety.ps1` | The real `safety.cpp` and `HolonomicMotion.cpp` with only the hardware layer replaced: tokens, stop, disable, mode change, watchdog, NaN, motor failure, timeout, busy drive, odometry epoch change and loop delay |
| `node simulator/test_override.js` | Inventory, ownership, stacks, visibility, autonomous snapshot, final score, midfield, wall, dimensions, match loads and manipulation |
| `test_holonomic_drive.js` / `test_physics.js` | 3 motion checks and 7 regression checks of the approximate physics. The old JS PID in the last test is a library check, not proof that it matches the active C++ loop |
| ARM | All 39 C++ translation units compile and link with the PROS libraries. Nothing was uploaded to the V5; the verification ELF does not replace the normal `pros make` / `pros upload` process |
| Browser | WASM loading, 2D/3D and robot.stl; a Bézier path finished through the UI with status `Settled`; manual actions are blocked in autonomous and drop/pickup work in driver control. Clipping in a narrow panel was fixed: at a 757 px width the document is also 757 px wide with no horizontal scroll. Exact robot CAD geometry does not take part in collisions: approximate outlines are used |

Selected model results, coordinates in inches, angle in degrees:

| Test | True model pose | Odometry estimate |
|---|---|---|
| Straight 24″ | `(0, 24.048, 0)` | `(0, 24.116, 0)` |
| Spline to `(20,40,45°)` | `(19.638, 40.279, 44.138°)` | `(20.047, 40.129, 44.988°)` |
| Bézier to `(24,24,90°)` | `(23.543, 24.330, 88.204°)` | `(23.990, 24.108, 89.772°)` |
| Straight 24″, low traction | `(0, 22.133, 0)` | `(0, 24.115, 0)` |

The last case shows an important limitation: `Settled` refers to the sensor estimate. With slip, the true position error here is about 1.87″, although odometry considers the target reached. Unobservable slip cannot be fixed by choosing LQR instead of PID. With zero traction the body stays in place while the encoder estimate keeps moving — this is also tested.

## Kept but not connected or certified by this check

LemLib PID/LQR/Hybrid, the earlier PedroFollower/GVF, the differential LTV, EKF, MCL, FLC, ColorSorter and the other original libraries remain in the project. Their presence does not mean they are active. The X-drive competition flow does not select PID/LQR/Hybrid. The joystick and the single-wheel test are manual mode and diagnostics.

The current configuration uses motor encoders, without an IMU or external tracking wheels; the intake and other mechanisms are not connected in `main.cpp`. Old mobile-goal autonomous routines are kept in the simulator as historical scenarios, not declared working Override autonomous routines. The old libraries did not receive a full independent audit in this check.

The kS/kV/kA, Q/R, PI, traction, mass and battery parameters need measurements on the robot. Actual DARE time on the V5, task jitter and the physical stop when a motor connection is lost were not measured. Browser odometry does not fully reproduce PROS delays, filtering and scheduling. The controller uses the same C++ code, but this is not a full firmware emulation.

## Next steps towards a full simulation

1. Check the physical robot's ports, wheel directions, geometry and odometry sources. Measure kS/kV/kA, PI, traction, mass, inertia, delays and voltage under load; attach logs.
2. Add replay of those logs to the tests and measure the LTV loop duration on the V5. Connect external sensors only after confirming their ports and orientation.
3. Load the official field CAD and verify dimensions/contacts. The official Field CAD link was found, but automatic download in this environment returned HTTP 403; the current geometry is built from drawings and remains approximate.
4. Implement objects as physical bodies: contact with the robot and each other, free fall, tipping, removing/carrying stacks, real geometry and manipulator limits.
5. Connect the autonomous program of each of the four robots and the real mechanisms. Currently the UI controls one selected robot and the others are static; the first 15 seconds of the game mode do not run the four robots' autonomous routines.
6. Complete the Override rule checks against the authoritative PDF: possession/zone limits, protections, violations and exceptions. Add game replays with expected scores and independent reconciliation. The software scorer is not a full referee yet.
7. Run Java Pedro Pathing / C++ differential tests on a shared set of curves and document the supported part of the API. Add a full simulated hardware layer if the whole PROS application, not just the shared core, needs to run.

Game sources: [official web manual](https://www.vexrobotics.com/override-manual), [authoritative PDF](https://link.vex.com/docs/26-27/v5rc/game-manual), [Field CAD](https://link.vex.com/docs/26-27/v5rc/field-cad). The web manual was used as the FO-1/FO-2 source; completeness of all rules is not claimed.

## Python files and the previous chat

The files `scratch/work/patch_*.py` are one-off scripts that modified sources. In particular, `patch_main.py` replaces `main.cpp` entirely; this explains the technical method of the simplification but does not prove which model ran the script. `get_zig.py` / `resume_zig.py` relate to downloading the compiler. They are not the game engine and are not needed to run the finished simulator. These scripts were not run again.

The supplied messages and context files were read. Earlier changes cannot be reliably split between Astra and Luna by files/commits alone: that requires reliable records of specific runs. This report describes verifiable code changes without attributing unknown actions to a model.


## Field and interaction improvements — 20 September 2026

[Robostem](https://sim.robostem.org/) was used as a visual reference for the scene and practice mode. Object placement and orientation were checked against FO-2 of the [official web manual](https://www.vexrobotics.com/override-manual). Robostem code and models were not copied; the geometry here is procedural.

- `override.js`: 16 lying Pins now point their coloured half outward; another 16 starting Pins are linked to supporting Cups. The shared `objectPose` function accounts for 3.25″ nesting, stacks on goals and held objects. Placing a preload now changes its location to `goal`, so it no longer disappears from the scene. Ownership is synchronized with object status. A Cup holding a Pin is carried as a pair when both slots are free; the Pin can be taken separately.
- `override-geometry.js`, `simulator.js`: contact of a rotated rectangle with the wall, circular approximations of goals and other robots' rectangles. The correction affects the body and its velocity while keeping independent encoder integration. The goal base radius of 4.25″ is an approximation; free Cups/Pins, loaders and mechanisms are not yet dynamic obstacles. Contact does not model angular impacts, deformation or pushing other robots.
- `override-view.js`: Pin/Cup dimensions are consistent between 2D and 3D, lying Pins point along the field, Cup halves are transparent/opaque, and stacks nest according to shared poses. Light grey tiles and loader posts make the scene easier to read. This is not an exact CAD copy.
- `index.html`, `override.js`, `simulator.js`: added practice without a time limit, with immediate manual control and no autonomous bonus. The 15+105 second match is kept.

Checked: the new `test_override_geometry.js` (inventory/orientations/supports, grabbing a pair, placement, preload, practice longer than 120 seconds, rotated contacts and full nesting of an obstacle in the body, a real simulator loop at the wall); the previous `test_override.js`; 17 C++ cascade scenarios, 3 holonomic tests, 7 approximate-physics checks and configuration freshness. In the browser, 3D, `PRACTICE ∞` and the **Put down** action were checked; no JavaScript errors or warnings were observed. C++ was not changed in this round, and the ARM build was not repeated.

In the wall-contact test, commanding a 24″ drive from Y=58″: body Y=64.95″, odometry Y≈68.45″; result `TimedOut`. This checks the difference between the body and the encoders during contact, not a measurement of the real robot. Full dynamics of falling objects, the manipulator, four autonomous robots and full refereeing remain next steps.


## Game loop — 25 September 2026

Added free fall and object contact, a lift with finite speed, nesting through physical release, breaking of a falling pair and loss of support. Four robots run autonomous routines on independent instances of the unchanged C++ cascade. In the demo, two preloads score 5 points each; after an equal autonomous period with the bonus the score is 11:11. Switching the controlled robot keeps the states of all four.

Score freezing was fixed: after 0:00 objects get up to five seconds to settle. Zero velocity of a just-released object does not count as rest while it is in the air. The end of autonomous stops lift tasks together with the drive; a DQ blocks both. Typing in the referee log does not drive the chassis, and losing focus releases keys.

Added autonomous routine selection, visualization of the four lifts, the referee log and JSON match-record export. Full implementation status, tests and remaining limitations are described in [SIMULATOR.md](SIMULATOR.md). Earlier notes about three stationary robots and no falling objects refer to the previous stage. Full automatic refereeing, exact CAD contacts and the collapse of scored stacks remain unfinished; the presence of the log does not replace their implementation.


Check for this stage: the full two-minute match and both new groups of physics tests passed; the 17 C++ cascade scenarios, 3 holonomic checks and 7 plant regressions are kept. In the browser all four programs showed `Complete`, after 0:00 the score was frozen at 11:11 and the lift rejected new commands. C++ sources and WASM were not changed; a repeated ARM build was not needed.
