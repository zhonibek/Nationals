# Nationals-work3

**Nationals-work3** is a VEX V5 robotics project and experimental robotics-AI workspace. It brings together PROS firmware, a browser-based VEX game simulator, the original Simulator/iraLIB motion environment, and RoboProof tools for inspecting and experimenting with robot movement.

The repository contains both working engineering components and research-stage features. A simulator pass, successful training run, or AI-generated recommendation is not by itself proof that a learned policy is better or safe on a physical robot. The current controller remains the default; experimental learned movement must pass the documented qualification gates before promotion.

## Contents

- [Quick start](#quick-start)
- [Project architecture](#project-architecture)
- [RoboProof and machine learning](#roboproof-and-machine-learning)
- [VEX Override simulator](#vex-override-simulator)
- [VEX V5 robot firmware](#vex-v5-robot-firmware)
- [Checks and builds](#checks-and-builds)
- [Limitations and safety](#limitations-and-safety)
- [Project map](#project-map)
- [Further documentation](#further-documentation)

## Quick start

### Open the VEX Override simulator

From the repository root, run this in PowerShell:

```powershell
.\tools\start-simulator.ps1
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765). The simulator needs to be served over HTTP; opening its HTML file directly with `file://` will not load the WebAssembly controller. The project includes the built `control.wasm` module. Three.js is loaded from a CDN for the 3D view, so that view requires an internet connection.

### Open RoboProof

From the repository root, run:

```powershell
node roboproof/cli.js motion-check
node roboproof/server.js
```

Open [http://127.0.0.1:8766](http://127.0.0.1:8766) and select **Motion lab**. The dashboard reads saved experiment artifacts; opening or refreshing it does not start model training. The `motion-check` command checks the motion-learning integration and is not a training run.

To run the deterministic RoboProof demo instead:

```powershell
node roboproof/cli.js demo
```

The simulator and RoboProof dashboard are separate local web applications and use different ports.

## Project architecture

### Robot motion stack

The configured holonomic X-drive uses a cascaded controller:

```text
path/reference → LTV-LQR motion control → four-wheel kinematics
              → wheel-speed PI + feedforward → motor voltage commands
```

The controller follows generated motion references, including quintic splines and Bézier paths. Its portable C++ core is shared with the browser simulator through WebAssembly, while the RoboProof motion environment connects bounded reference actions to the original Simulator and iraLIB controller. The physical firmware obtains sensors, handles timeouts and cancellation, and writes motor commands through PROS.

The current autonomous X-drive path uses one cascaded controller; it does not switch between separate PID, LQR, and hybrid autonomous modes. The default wheel-speed derivative term is zero, so the configured wheel loop is PI plus `kS`/`kV`/`kA` feedforward. Legacy functions named `autoPIDDrive` and `autoPIDTurn` remain compatibility wrappers and do not select a different controller.

### RoboProof applications

RoboProof is the project’s local dashboard and experimental robotics-agent workspace. Its modules have different responsibilities and different levels of readiness:

- **Motion lab** — inspect controller and learning experiments, compare recorded outcomes, and replay selected actions in the original Simulator.
- **Tactical AI** — provide rules-grounded, conditional recommendations from a verified game-rule set and an explicitly selected simulator snapshot. Recommendations do not execute robot actions.
- **Perception preparation** — validate a future object-location request format and calibration inputs. Camera inference and automatic movement from a visual target are not implemented.
- **Nemotron / Cosmos integrations** — optional model-runtime workspaces. They require their documented model/runtime setup and are separate from the embedded movement controller; their presence does not mean that model weights are installed or that inference is currently available.

There is not yet a unified voice assistant that safely turns arbitrary natural-language requests into plans, obstacle-aware navigation, game actions, and verified physical motion.

## RoboProof and machine learning

### What the movement learner does

The motion learner is a small PyTorch actor-critic reinforcement-learning system, not a general-purpose language model. It reads a defined robot-state/reference observation and emits bounded reference adjustments. Those actions go through the existing Simulator and iraLIB/C++ control stack; the learner does not directly output motor voltages or replace the underlying controller.

The repository contains PPO and TD3 research paths, training-data and checkpoint contracts, experiment/replay tooling, evaluation gates, and optional architecture settings. Two hidden layers remain the baseline. A deeper or wider network—including an optional ten-hidden-layer experiment—is an architecture choice, not evidence of better control. Larger networks need their own training and evaluation; they do not inherit qualification from the baseline.

### Current status and interpretation

- Gradient updates, saved checkpoints, experiment replay, and implementation tests demonstrate that training software runs; they do **not** demonstrate improved movement.
- Learned movement remains experimental and is not the default policy. No candidate should be described as qualified unless the exact source/runtime-bound evidence passes the independent gates.
- The strict improvement gate requires each of three independent training seeds to outperform both controller-only and untrained baselines in success, have zero regressions on baseline-success cases, avoid per-world contact increases, and improve mean time or effort by at least 5%. A training reward increase, reduced loss, one good seed, or one successful replay is insufficient.
- The learner does not currently observe camera images, game objects, or obstacle maps; it does not learn a complete game-scoring strategy or manipulation policy. Its scope is bounded movement-reference adaptation around the existing controller.
- Trained actor weights and run outputs are artifacts, not source code. They are not automatically included in this Git repository. Follow the [actor deployment guide](docs/GVF_MODEL_DEPLOYMENT.md) and verify the artifact’s source/runtime compatibility before using it.

### Remote training policy

All training, fine-tuning, imitation learning, large-scale teacher-data collection, and mass simulation campaigns must run only on an explicitly approved remote server or private cloud notebook. Do not use this computer’s CPU/GPU as a fallback when a remote runtime is unavailable. Before starting a run, verify the actual allocation, device, runtime, approved quota/cost, source, and Simulator/iraLIB identity. Preserve checkpoints, complete logs, failures, exposure records, and result hashes.

Local work is limited to code editing, the dashboard, packaging/provenance checks, and lightweight focused tests. Do not put credentials or model tokens in Git. See [`AGENTS.md`](AGENTS.md), the [motion learner guide](roboproof/motion_learning/README.md), and the [remote training custody rules](docs/REMOTE_TRAINING_CUSTODY.md) for the detailed execution contract.

## VEX Override simulator

The `simulator/` application is a browser-based VEX Override game environment with 2D and 3D views. It includes a game clock, autonomous period, manual driver controls, robot programs, item interactions, scoring, event logs, and JSON protocol export.

### Run a game

1. Start the simulator using the [PowerShell command above](#open-the-vex-override-simulator).
2. Select **Training ∞** for unlimited manual practice or **Match 2:00** for a timed match.
3. Press **Override** to start the selected mode.
4. Select a robot and use the on-screen controls or keyboard controls shown by the simulator.

In a match, the autonomous period runs the four selected autonomous programs independently before driver control. Each robot has its own C++ controller instance and motor/odometry state. Pressing **Override** starts the selected game mode again. **Run** launches the separate autonomous-route test stand.

The simulator models the Override field and game loop, robot movement and contacts, scoring, and Cup/Pin handling. Cups and Pins can be carried in separate slots; a Cup with a Pin nested on top can be moved and placed as a pair when both slots are available, or the Pin can be removed separately. The lift moves through its bounded range at a finite speed. Pickup depends on fork-to-object horizontal distance and height; goal placement requires alignment, and points are awarded when the objects enter the scoring opening. A free drop falls and may tip. The simulator can export programs, paths, events, and score as JSON.

### Simulator fidelity

This is a useful development and demonstration simulator, not a validated digital twin or a complete official-game referee. It does not emulate PROS, the V5 scheduler, or the complete robot firmware. Motor response, traction, battery behavior, collisions, and field contacts are approximations and have not all been identified against the physical robot. Goal scoring uses simplified geometry; already-scored stacks are currently fixed. Some disputed rules and intent require human judgment. Consult [the simulator guide](docs/recovery/SIMULATOR.md) for implemented behavior and known limitations.

## VEX V5 robot firmware

The firmware targets a VEX V5 robot using PROS. The active `main.cpp` configuration uses the holonomic X-drive, four motor encoders, the LTV cascade, and a motor monitor. The current configuration does not connect the IMU, external tracking wheels, intake, or other mechanisms. Check the actual robot wiring before changing ports or uploading firmware.

### Configuration and units

- [`config/robot.json`](config/robot.json) contains the current port and geometry configuration.
- After changing the JSON configuration, regenerate the C++ configuration with `node tools/generate-config.js`.
- [`include/autonomous.hpp`](include/autonomous.hpp) defines the motion API.
- [`include/subsystems/control/Cascade.hpp`](include/subsystems/control/Cascade.hpp) contains the portable cascade core.
- [`src/subsystems/control/HolonomicMotion.cpp`](src/subsystems/control/HolonomicMotion.cpp) connects sensing, timing, cancellation, and PROS motor output.
- The public motion API uses inches and degrees, with clockwise heading measured from `+Y`. The controller internally uses meters, meters per second, radians, and volts. Positive timeouts are milliseconds; `timeout=0` derives a limit from the motion profile.

The LCD route selector is available while the robot is disabled. It chooses among the cascade demonstration, holonomic skills route, and Bézier route; it does not choose a controller. The demonstration goes to `(24, 24, 90°)` and returns to `(0, 0, 0°)`. A chained route starts its next segment only after the previous one reports `Settled`.

In driver control, the configured shortcuts are: **A** cancel, **B** drive forward 24 in, **Y** turn 90°, **X** strafe 24 in, **Left** turn 180°, **Down** diagonal motion, **Right** Bézier, **R1** spline, **Up** individual-wheel test, and **L1** pose reset. Only reset the pose when the robot’s physical position is known.

The repository also contains LemLib, PedroFollower, EKF, MCL, FLC, ColorSorter, and differential-drive LTV code. Their presence does not mean they are all active in the current robot configuration. Older AWP/Goal Rush routes use previous game assumptions and are not ready-to-run Override autonomous programs.

## Checks and builds

Run focused checks from the repository root:

```powershell
node tools/generate-config.js --check
node tests/control-core.test.js
node simulator/test_override.js
node simulator/test_override_geometry.js
node simulator/test_override_dynamics.js
node simulator/test_override_match.js
node simulator/test_holonomic_drive.js
node simulator/test_physics.js
```

Additional RoboProof and learning tests are under `tests/`. Training tests and large simulation suites must follow the remote-only execution policy; do not run them locally as a fallback.

### Build the browser controller

Building the WebAssembly C++ module requires Zig 0.14.1:

```powershell
.\tools\build-wasm.ps1 -Zig C:\path\to\zig.exe
.\tools\test-safety.ps1 -Zig C:\path\to\zig.exe -Cache C:\path\to\cache
```

`control-build.json` binds the generated WASM to source hashes. The safety test rejects a stale build. The build creates a browser module; it does not upload firmware to a robot.

### Check the V5 firmware build

The ARM verification script accepts a PROS toolchain path and a separate build directory:

```powershell
.\tools\verify-arm-build.ps1 `
  -Repository . `
  -ToolchainBin C:\path\to\pros-toolchain\usr\bin `
  -BuildDirectory .\.cache\arm-check
```

This is a verification build and does not flash the robot. For the normal PROS workflow, inspect the configuration and use the appropriate `pros make` or `pros upload` command only when you intend to build or deploy to the physical robot.

## Limitations and safety

- A passing software test is not proof of physical-robot safety. Test hardware changes with the robot secured and follow your team’s VEX safety procedures.
- The browser game simulator and the motion-learning Simulator/iraLIB environment serve different purposes. The motion learner’s reward is not the official VEX game score, and learning movement does not automatically teach game tactics or object manipulation.
- The current motion learner has no camera/obstacle input or unrestricted motor command. It cannot safely interpret arbitrary voice commands or autonomously drive to a visual point without additional perception, planning, and safety work.
- Analytical controllers remain the default. Do not activate or promote a learned policy based only on a training curve, score, single episode, or model-generated explanation.
- Do not claim that an optional Nemotron/Cosmos model, camera pipeline, or cloud GPU is operational until its actual runtime and output have been verified.
- Checkpoint files, private training data, API tokens, SSH keys, and cloud credentials must not be committed to Git.

## Project map

| Path | Purpose |
| --- | --- |
| `src/` | PROS robot firmware and controller implementations |
| `include/` | Firmware APIs, robot configuration, and portable control headers |
| `config/` | Robot ports and physical configuration |
| `simulator/` | Browser-based VEX Override simulator and WebAssembly control module |
| `roboproof/` | RoboProof dashboard, simulation/learning orchestration, AI modules, and model integrations |
| `roboproof/motion_learning/` | Python PPO learner, actor/checkpoint contracts, and experiment support |
| `roboproof/ml/` | Separate ML data, model, job, and validation utilities |
| `tests/` | Controller, simulator, RoboProof, and learning tests |
| `tools/` | Configuration, simulator, WASM, packaging, and validation scripts |
| `docs/` | Architecture, experiments, recovery notes, plans, and operating procedures |
| `assets/` | Project assets |

## Further documentation

- [Motion-learning implementation and limits](roboproof/motion_learning/README.md)
- [Integrated motion-learning plan](docs/MOTION_LEARNING_PLAN.md)
- [Motion episode and simulator contract](docs/MOTION_EPISODE_CONTRACT.md)
- [GVF integration notes](docs/GVF_INTEGRATION.md)
- [Saved GVF actor deployment constraints](docs/GVF_MODEL_DEPLOYMENT.md)
- [Tactical AI plan](docs/TACTICAL_AI_PLAN.md)
- [Perception preparation](roboproof/perception/README.md)
- [Nemotron setup and restrictions](roboproof/NEMOTRON.md)
- [Simulator implementation and limitations](docs/recovery/SIMULATOR.md)
- [Recovery report and verification evidence](docs/recovery/REPORT.md)
- [Development progress and experiment history](docs/progress.md)
