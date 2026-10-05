# IRAlib / Nationals

C++ robot control and simulation for **VEX V5**, built with **PROS 4.2.2**, **Eigen**, and **LemLib foundations**.

This repository brings together robot configuration, autonomous routines, control and localization modules, diagnostics, and a browser simulator. It is an engineering project for studying and developing competition robot behavior. Hardware timing, accuracy, and reliability should be evaluated on the configured robot.

## Review the project in five minutes

1. Read [`src/main.cpp`](src/main.cpp) for motor/sensor configuration, competition lifecycle functions, autonomous routines, and driver control.
2. Explore [`src/subsystems`](src/subsystems) for the control, estimation, and diagnostic modules.
3. Run the simulator checks below, then open the browser simulator.
4. Read [`docs/TUNING_GUIDE.md`](docs/TUNING_GUIDE.md) for calibration and tuning procedures.

## Components

| Area | Implementation |
| --- | --- |
| Motion control | PID and LQR controllers, LTV path follower, velocity controller, fuzzy gain adjustment |
| Trajectory generation | Quintic spline generator |
| Localization | Odometry, extended Kalman filter, particle-filter modules, sensor-based pose resets |
| Diagnostics | Motor connection/temperature monitor and optical color sorter |
| Competition integration | Autonomous routine dispatcher, holonomic driving, controller-based calibration tests |
| Simulation | JavaScript robot/field model, visualization, and headless validation scripts |

The presence of a module does not mean every autonomous routine uses it. Follow the call sites in `src/main.cpp` and the subsystem implementations when assessing integration.

## Run the browser simulator

From the repository root, using Python:

```bash
python -m http.server 8000 --directory simulator
```

Open **http://localhost:8000** in a browser. The viewer loads Three.js and related scripts from public CDNs, so it requires internet access for those dependencies. A local HTTP server also helps the browser load the robot model reliably.

The simulator is a JavaScript model. Its results do not establish numerical equivalence with the C++ firmware or measured performance on a physical robot.

## Run validation scripts

Requirements: Node.js; no npm packages are needed for these two scripts.

```bash
node simulator/test_physics.js
node simulator/test_holonomic_drive.js
```

Both scripts completed successfully during a local review on **5 October 2026**:

- Physics suite: seven checks covering model behavior, state finiteness, motor limits, inertia changes, filtering, controller behavior, and wall collision.
- Holonomic suite: three trajectory checks covering translation while rotating, circular motion, and lateral zig-zag motion.

These are simulator checks. Firmware compilation and physical robot operation were not verified in that review.

## Build the firmware

Install the official PROS toolchain for VEX V5 and run from the repository root:

```bash
pros make
```

The project configuration specifies the PROS **4.2.2** kernel. Review motor ports, polarity, sensor configuration, geometry, and control gains in `src/main.cpp` before using the firmware on hardware. The browser simulator lives outside `src/` and is separate from the firmware build.

## Repository map

```text
src/main.cpp                 Robot configuration and competition routines
src/subsystems/              Control, localization, and diagnostic modules
include/subsystems/          Subsystem interfaces
src/lemlib/                  LemLib-based control and odometry code
include/Eigen/               Bundled numerical computing headers
simulator/                   Browser viewer and validation scripts
docs/TUNING_GUIDE.md         Calibration and tuning procedures
project.pros                 PROS project configuration
```

## Current validation limits

The repository needs a hardware results record to substantiate claims about tracking error, settling time, loop frequency, and competition reliability. Useful evidence would include the tested commit, hardware configuration, repeated trials, logs, and a robot demonstration video. Simulation alone does not establish these results.

## Attribution and license

This project incorporates [LemLib](https://github.com/LemLib/LemLib), [Eigen](https://gitlab.com/libeigen/eigen), and the [PROS](https://github.com/purduesigbots/pros) ecosystem. The `lemlib` directories contain foundational library code; they should not be presented as wholly original work. See the repository [MIT license](LICENSE) and bundled dependency notices for their respective terms. Preserve upstream copyright notices.
