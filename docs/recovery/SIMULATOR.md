# Nationals-work3 game simulator

Implementation status: 25 September 2026. The motion controller remains the persistent cascade LTV-LQR → four velocity PIDs with feedforward. The new mechanics belong to the browser model; the PROS firmware and its hardware configuration were not changed in this work.

## How to check

Run `tools/start-simulator.ps1` and open http://127.0.0.1:8765/. Choose **Match 2:00**, optionally expand the settings of the four autonomous routines, and press Override. By default red-1/blue-1 drive to their own goals, lift the preload and release it; red-2/blue-2 move slightly away from the wall. These are test game programs, not a ready competition strategy.

After 15 seconds, manual control is available for the selected robot. WASD/QE — translation, Left/Right arrows — turn. A lift step of ±2″ changes the target height instead of teleporting the mechanism. Switching the selected robot keeps every robot's pose, encoders, motors and mechanism. In practice mode, autonomous routines do not run and there is no time limit.

To grab, the front fork must be within 3″ of the object's centre horizontally and within 3″ vertically. These are model grabbing tolerances, not the dimensions of an official mechanism. To place, align within 1.1″ of the goal axis and set the height at the top of the stack. **Place in goal** releases the object; nesting is determined when it falls. A robot can hold one Pin and one Cup. While carrying a nested pair, both slots are occupied.

An empty Cup on the floor can accept a falling Pin. When a Cup tips over, its Pin loses support. A hard fall separates a pair. After 0:00 the drive and mechanism no longer accept commands; passive object motion continues until it stops or for five seconds, then the score is frozen.

## Architecture

| File | Purpose |
|---|---|
| `simulator/control-runtime.js` | Creates independent WASM instances of the same production C++ controller |
| `simulator/override-autonomy.js` | Four motor models, separate command queues, exact period boundaries, observed-robot selection, match record |
| `simulator/override-dynamics.js` | Gravity, finite velocities, approximate object contacts, floor impact, tipping, loss of support, nesting |
| `simulator/override.js` | Inventory, mechanism, action availability, scoring and referee rulings |
| `simulator/override-view.js` | 2D/3D rendering of shared object poses and the four lifts |
| `simulator/simulator.js` | Connects the game loop with the drive model and the interface |

`OverrideFleet` executes the steps `pose`, `drive`, `strafe`, `turn`, `spline`, `bezier`, `diagonal`, `lift` and `game`. The next step's motion is allowed after `Settled`; an error stops that robot's program. At 15.000 s unfinished motions and lift tasks are cancelled. For tests and the API, `startGame(mode, choices)` accepts a map from robot IDs to program names or step arrays. The UI offers verified demonstrations.

The JSON match record contains programs, the final state, grab/place events and referee rulings; pose, odometry, voltages and status for each robot are sampled every 0.5 s, up to 600 samples. It is a diagnostic log, not a bit-exact replay of the whole firmware or all contacts.

## Rule coverage

Source: [VEX Override v2.0 web manual](https://www.vexrobotics.com/override-manual), checked during implementation; the [authoritative PDF](https://link.vex.com/docs/26-27/v5rc/game-manual) takes precedence. Rules are not reproduced verbatim here.

| Area | Implemented | Limitation |
|---|---|---|
| SC1 | Command stop, up to 5 s of passive motion, score freeze | Disputed states at the boundary are left to the referee |
| SC2–SC6 | Alternating nesting, half visibility, ownership by Toggle/location, midfield | Partial/angled nesting and continuous Toggle contact are simplified |
| SC7–SC8 | Autonomous score snapshot, bonus, AWP, award transfer by a confirmed ruling | Borderline contacts require checking the log |
| SG1–SG6 | Starting layout, preload, model dimensions, one Pin + one Cup | Real robot inspection and intentional ejection are not modelled; walls retain objects |
| SG7–SG8 | Boundary-crossing marker; 28 initial shared objects; tracking of illegal grabbing of the other side's objects | Not all indirect contacts and exceptions are detected automatically |
| SG9–SG10 | Button placement on the opponent's goal is blocked; objects cannot be removed from locked stacks | Manipulator collisions with a protected stack are not reproduced |
| SG11 | Own loader, driver period, object drop through the loader | Contact with operators' hands and the loader body are simplified |
| SG12 | No placement in the centre goal during the endgame; log of actual late nesting | Disputed episodes are judged by the referee |
| SG13 / GG / S | A rule, reason, warning/minor/major and robot DQ can be recorded | Intent, trapping, tipping and load-zone interference have no complete automatic classifier |

The referee log can explicitly transfer the autonomous bonus and, with the corresponding reason, the AWP. A DQ disables the drive and actions of that robot. Points earned by objects are not turned into arbitrary penalty deductions. Final tournament WP/SP and DQ consequences for qualification/elimination are not computed. Therefore this is **not a complete automatic referee**.

## Physical fidelity

Free objects use inches/seconds, gravity of 386.09 in/s² and a step of up to 5 ms. Horizontal contacts are circular approximations; the robot body is a rotated rectangle. Pair impulses use equal effective masses. Tipping and loss of support are a reduced model. Exact contact geometry of lying objects, shape friction, rotation about all three axes and elastic deformation are not implemented.

Stacks already nested on goals are locked; they do not collapse on impact. The field is not a full CAD rigid-body world. The lift is a kinematic actuator with a speed limit and no motor/load model of its own. All four robots use the parameters of the single Nationals drive. Robot-to-robot contact does not model pushing battles or chassis tipping.

## Checks

`test_override_match.js` runs a full two-minute match: separate WASM memories, two physical preload placements, 15/120 s transitions, robot switching without reset, commands after the end, lift cutoff, invalid dt, DQ and match-record serialization.

`test_override_dynamics.js` checks lift speed and limits, falling against the gravity formula, impact/tipping, grabbing only in front of the robot, nesting on a goal and on the floor, breaking of a falling pair, loss of support, object-robot contact, walls, the scoring grace period, the hard five-second limit and a late referee ruling.

The older `test_override.js` and part of `test_override_geometry.js` use `physical:false` for isolated rule fixtures. That is not the browser mode: the new integration tests run the real game physics and C++ controllers. The previous 17 cascade scenarios, holonomic/plant checks and WASM freshness checks are kept.
