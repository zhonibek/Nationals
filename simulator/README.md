# Nationals-work3 simulator

Run from the repository root: `powershell -File tools/start-simulator.ps1`, then open http://127.0.0.1:8765.

`control.wasm` executes the same C++ LTV-LQR → PID + feedforward cascade that the competition flow uses. Controller modes are not switched. `Competition` in the routine list replays the current `autonomous()` sequence; Bézier and quintic paths use the production C++ code.

Tests: `node tests/control-core.test.js` and `node simulator/test_override.js` from the repository root. Detailed instructions and limitations are in the [project README](../README.md) and the [recovery report](../docs/recovery/REPORT.md).

Controls: W/S — forward/backward; A/D — strafe; Left/Right arrows — turn; Space — pause. In the game, hold E to raise the lift and Q to lower it; release the key to stop. In the separate motion test, Q/E remain extra strafe keys. **Step** runs one 10 ms step. **Start game** enables the normal field, not a tutorial scene. Practice starts immediately; in a match, the autonomous routine runs for the first 15 seconds. You can select any of the four robots; the others stand still. Each robot holds at most one Pin and one Cup, and grabbing requires proximity and the right height; this is an abstraction of the manipulator.

Chassis physics and object geometry are approximations. The 3D view loads Three.js from a CDN; the 2D view and the controller run locally. Ports and dimensions come from the generated `robot-config.js`. Ground-truth pose is used only for rendering and collisions; the controller receives only encoder odometry and wheel velocities.

Game data source: [VEX manual v2.0](https://www.vexrobotics.com/override-manual), field drawings FO-1/FO-2. The [official PDF](https://link.vex.com/docs/26-27/v5rc/game-manual) is authoritative. CAD for game elements, detailed stack dynamics, full firmware emulation and automatic refereeing are not finished.

## Game mode and scoring — 2026-10-04

- **Start game** enables object contacts and the manipulator. **Run** separately tests chassis motion and does not start the game. Continuing a game no longer starts a route or resets the field.
- There are no separate tutorial buttons. Practice allows free grabbing by default; preload Pins taken from the start return to their alliance's stock without changing the rest of the field. You can enable **Preload Pin at practice start**; an official match always keeps the original preloads.
- WASD — move, arrows — turn, E/Q — lift up/down. F — take the nearest available Pin, Cup or Cup + Pin stack; next to a Toggle the same F key switches it. The hint and the single action button show the selected action; if both are available, the one closest to the robot wins. G — place, R — release, T — flip. Physical key positions work with any keyboard layout. In a match, a held preload Pin must be placed or released before grabbing another stack.
- Hints show held objects, distance to the fork, height error and the allowed height of the next layer. Unavailable actions are disabled. The lift moves at a limited speed; grabbing does not teleport distant objects.
- A Pin falls and counts only after it lands in a goal. Stacks alternate Pin/Cup. The scoreboard explains visible halves, yellow ownership, robots in midfield and the autonomous bonus; the manual calculator does not change the field. The JSON match record includes the score breakdown.
- The 2D view can show contact proxies, including the full length of lying parts. Physics handles object-to-object contacts, velocity transfer from the robot, gravity, floor friction and walls. A spatial grid limits pair checks to neighbouring cells.
- Opaque rendering is on by default for readability. Physical transparency is not a collision switch, and the visual toggle does not change the half-visibility rules used for scoring.

Dimensions, sources and accuracy limits: [OVERRIDE_ELEMENTS.md](../docs/OVERRIDE_ELEMENTS.md). This is an improved approximate simulation, **not a full copy of real field physics**: game-element CAD is not imported, masses, friction and impacts need measurement, and collisions with held objects and the collapse of placed stacks are not fully modelled yet.

## Robot CAD and performance — 2026-10-05

The robot uses a lightweight version of the supplied `main.glb`: 166 MB → 5.07 MB, 117,612 scene primitives → 366, 6.40 M triangles → 407 k. The original is unchanged. Game-element CAD is still not imported. This is a visual model; it does not automatically provide the real robot's grabbing mechanics, mass or collisions.

Reload the page and open 3D: `robot.preview.glb` loads automatically. Do not import the heavy original; files above 32 MiB are rejected before reading. The default quality is **Fast · no shadows · 30 FPS**; the measured FPS and full-field render cost are shown next to it. CAD is not loaded in 2D. If needed, use the procedural model or stay in 2D. Physics and the C++ controller keep their 10 ms step.

Preparation, sources, hashes, limitations and reproduction: [ROBOT_CAD.md](../docs/ROBOT_CAD.md). Real-browser smoothness and appearance have not been checked yet.

## Cup grabbing — 2026-10-05

Candidate selection is fixed: a Cup holding a Pin no longer hides a neighbouring free empty Cup. While the Pin slot is occupied an empty Cup is allowed, but a Cup + Pin stack requires placing or releasing the preload first. Instead of a generic message, the specific reason for a refusal is shown. The limit of one Pin and one Cup still applies.

In **Auto** mode a single F / **Take** works with all object types without switching the list. A held Cup + Pin stack is lifted and moved together; G/R release it as a whole. To take only the top Pin, choose **Pin only**. For a Match Load you can explicitly choose Pin or Cup (Auto loads a Pin). Detailed hints for both types are in the expandable block. The robot CAD does not define real grabbing kinematics.

To check: reload with **Ctrl+F5 → Practice ∞ → Start game** and keep grabbing on **Auto**. Bring the front fork to an object, wait for the **Can take** hint and press F. E/Q move the lift smoothly within 0–40″; **Height for goal** sets the height of the next layer instead of teleporting the mechanism. A Cup is placed on top of a Pin; an empty goal accepts a Pin first. Points appear after the object physically lands in the goal. Released high above the floor, a stack can break apart on impact — it is not glued together. Prepared scoring scenarios exist only in tests, not in the user interface.

## Score calculator — 2026-10-05

A compact scoreboard is pinned to the bottom of the window and links to the full calculator. Both show the same score breakdown from the game engine, including the frozen final score; the live value is explicitly marked preliminary. Visible Pin halves are counted, not whole parts; a Cup on its own scores 0 points.

The manual calculator recalculates on load, on number input and when the bonus changes. Empty, fractional, negative and oversized values produce an error and clear the old result instead of becoming zero. The arithmetic formula is shown separately. The red/blue buttons copy a snapshot of that alliance's game values; **Reset calculation** clears only the manual fields. These actions do not change the field, ownership or the game score.
