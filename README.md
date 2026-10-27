# Nationals Physical AI

> 🏆 **Nebius × NVIDIA Global AI Hackathon 2026 — Physical AI track.** Full write-up, architecture and feedback: [**HACKATHON_POST.md**](HACKATHON_POST.md) · License: [MIT](LICENSE) · Plan and progress: [docs/HACKATHON_PLAN.md](docs/HACKATHON_PLAN.md)

**Don't train robot AI from scratch — give it the tools every robotics programmer already knows.** A C++20 control cascade (LTV-LQR → wheel PID, guided by a vector field) drives our VEX V5 X-drive robot, and the *same* C++ runs in a browser simulator of the V5RC Override field through WebAssembly. A small residual network may only nudge the controller's reference, never the motors. **NVIDIA Nemotron on Nebius Token Factory** sits on top as a tool-calling coordinator: it plans motion tasks and plays whole matches by choosing legal actions that the game's rules engine checks and scores. **RoboProof** records the evidence for every claim, including the negative results.

## Contents

- [Quick start](#quick-start)
- [NVIDIA Nemotron and Nebius Token Factory](#nvidia-nemotron-and-nebius-token-factory)
- [Trained models](#trained-models)
- [Project architecture](#project-architecture)
- [RoboProof and machine learning](#roboproof-and-machine-learning)
- [VEX Override simulator](#vex-override-simulator)
- [VEX V5 robot firmware](#vex-v5-robot-firmware)
- [Checks and builds](#checks-and-builds)
- [Limitations and safety](#limitations-and-safety)
- [Project map](#project-map)
- [Further documentation](#further-documentation)
- [License, attribution and originality](#license-attribution-and-originality)

## Quick start

**Requirements:** [Node.js](https://nodejs.org) 20.3 or newer on Windows, macOS or Linux, and a current browser. Internet access is needed for the 3D view (Three.js from a CDN) and for Nemotron. Nothing else has to be installed to run the apps.

```bash
git clone https://github.com/zhonibek/Nationals.git
cd Nationals
npm start
```

`npm start` (also `start.bat` on Windows, `./start.sh` on macOS/Linux, or `node tools/start.js`) starts one local server on port 8766 and opens the portal at <http://127.0.0.1:8766/simulator/portal.html>. The same server serves the simulator (`/simulator/index.html`) and RoboProof (`/`). Options: `--port <n>`, `--no-open`.

**Connect NVIDIA Nemotron (optional, needs a free Nebius account):**

1. Create an API key in the [Nebius Token Factory](https://tokenfactory.nebius.com) console (API Keys).
2. Click the **Nebius Token Factory** badge at the top of the portal and paste the key. The server checks it against Token Factory first and stores it only in the Git-ignored `.cache/` folder; the browser never keeps it.

Or pass the key through the environment:

```bash
NEBIUS_API_KEY=your-key npm start            # macOS / Linux
```

```powershell
$env:NEBIUS_API_KEY = "your-key"; npm start  # Windows PowerShell
```

Without a key the simulator, the controller and the trained tactical policy all work; only the Nemotron features are unavailable.

**What to try:**

- **Simulator → Tactical AI match:** choose *Nemotron (Nebius Token Factory)* and run a match. Every decision is listed with Nemotron's reason; compare it with the *Loading-zone cycles* and *Field greedy* tactics.
- **Simulator → Tactical PPO optimizer:** type a task such as "score 3 match loads into the neutral goals, then park"; the trained PPO policy bundled in [`roboproof/models/tactical-ppo-seed736001`](roboproof/models/tactical-ppo-seed736001/README.md) carries it out in the tactical model.
- **Simulator → AI Programming:** describe a motion in plain language and run the generated route on the WebAssembly controller.
- **Simulator → Start game:** drive the robot yourself (WASD, arrows, E/Q lift, F take, G place).
- **RoboProof (`/`):** Motion Lab replays, the Nemotron workspace and the evidence views.

Run the tests with `npm test` (619 Node tests). The RoboProof command line (`node roboproof/cli.js motion-check`, `node roboproof/cli.js demo`) and the standalone simulator server (`tools/start-simulator.ps1`) are described in [roboproof/README.md](roboproof/README.md) and [simulator/README.md](simulator/README.md).

## NVIDIA Nemotron and Nebius Token Factory

**Models.** `nvidia/Nemotron-3_5-Lightning` is the default, used for fast per-step tool calls. `nvidia/nemotron-3-super-120b-a12b` was also evaluated on the same tactical task. Both are called at runtime through the OpenAI-compatible Nebius Token Factory API at `https://api.tokenfactory.nebius.com/v1`.

**What Nemotron does here** (it never outputs motor commands, code to execute or scores):

| Use | How | Code |
| --- | --- | --- |
| Motion task planning | Native tool calls `load_skill → get_motion_contract → prepare_reach_pose`; the task is validated and runs only after user approval; ambiguous requests end in `ask_clarification` | [`roboproof/nemotron.js`](roboproof/nemotron.js) |
| Match coordinator | Each decision, `choose_action` picks one candidate ID from a shortlist whose points and seconds the Override rules engine previewed; an invalid answer stops the match (fail closed) | [`roboproof/tactical-coordinator.js`](roboproof/tactical-coordinator.js), `POST /api/tactics/match` |
| Rules-grounded advice and chat | Reads the reviewed rule subset and a compact game snapshot; advice only | [`roboproof/tactics.js`](roboproof/tactics.js), [`roboproof/nemotron-chat.js`](roboproof/nemotron-chat.js) |

The client validates the endpoint (HTTPS Nebius hosts or loopback only), keeps the key server-side, and requires the configured model to be listed by `/v1/models` ([`roboproof/NEMOTRON.md`](roboproof/NEMOTRON.md)).

**Where Token Factory accelerated our workflow:**

| | Before (Nemotron on a laptop CPU) | Nebius Token Factory |
| --- | --- | --- |
| One motion-planning request | 153.7 s cold, 73.8 s warm | about 1 s per tool-calling turn |
| Endpoint check | — | 0.47 s |
| One tactical decision | not practical | ~0.6 s (Lightning), ~1.7 s (Super 120B) |
| A full driver period (~30 decisions) | not practical | ~18 s of model time (Lightning) |

That speed is what made per-decision match coordination possible, and switching between Nemotron sizes was a one-line model change instead of hosting each model ourselves.

**Result** (one driver period in the match model, robot timing measured on our robot, mean margin over seeds 101–102): loading-zone tactic 246, **Nemotron 3.5 Lightning 231**, field greedy 218, greedy over all actions 203, random 62.5. Nemotron Super 120B scored 203 on seed 101 and failed closed on seed 102 after one invalid tool call. Details: [HACKATHON_POST.md](HACKATHON_POST.md#tactical-ai-results).

**Other Nebius tools and services.** A Nebius Serverless AI Job for the tactical trainer is prepared ([`tools/nebius-tactical-job.sh`](tools/nebius-tactical-job.sh), [`tools/nebius-tactical-launch.sh`](tools/nebius-tactical-launch.sh)) but was not run: we had no Nebius compute credits, so training ran on a free NVIDIA Tesla T4 in Google Colab. Nebius AI Cloud compute was not used.

**Other NVIDIA technology.** Training used CUDA on an NVIDIA Tesla T4. An adapter for NVIDIA Cosmos-Reason2 ([`roboproof/cosmos`](roboproof/cosmos/README.md)) is prepared to describe recorded robot motion; its GPU inference has not run yet.

## Trained models

| Model | Parameters | Trained on | Result | Files |
| --- | ---: | --- | --- | --- |
| Residual GVF-TD3 actor, 36 → 10 × 512 → 3 | 2,384,387 | AMD Instinct MI300X, seed 50513, 564 episodes | research artifact, improvement not verified | [`roboproof/models/gvf-td3-mi300x-512x10-seed50513`](roboproof/models/gvf-td3-mi300x-512x10-seed50513/README.md) |
| Tactical PPO policy, 23 → 2 × 128 | 28,034 | NVIDIA Tesla T4, seed 736001 | held-out margin 201.5 vs loading-zone tactic 244.5 | [`roboproof/models/tactical-ppo-seed736001`](roboproof/models/tactical-ppo-seed736001/README.md) |

Full run records, including the failed and weaker runs, are in [docs/progress.md](docs/progress.md).

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
- **Nemotron / Cosmos integrations** — AI reasoning and task coordinator workspace. Nemotron is integrated with **Nebius Token Factory** (`nvidia/Nemotron-3_5-Lightning`), enabling sub-second natural language task planning and function-calling into the WebAssembly C++ controller.
- **AI Autonomous Programming Studio** — available directly inside the 3D Simulator: converts natural-language movement instructions into C++ PROS routines and executable 3D rollouts in real time.

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
- [Hackathon plan and progress](docs/HACKATHON_PLAN.md)

## License, attribution and originality

The project's own code is licensed under the [MIT License](LICENSE), Copyright (c) 2026 Kassymkhan and Nationals contributors.

- **Pedro Pathing** is an FTC path-following library originally written in **Java**. The code in [`include/subsystems/pedro/`](include/subsystems/pedro/NOTICE.md) and [`src/subsystems/pedro/`](src/subsystems/pedro/NOTICE.md) is an independent **C++ rewrite** for VEX V5 based on its design; no Java code is included. Pedro Pathing's BSD-3-Clause notice is retained.
- **LemLib** (MIT) is included in `include/lemlib/` and `src/lemlib/`; its copyright notice is kept in [LICENSE](LICENSE).
- **PROS** (MPL-2.0), **Eigen** (MPL-2.0) and **{fmt}** (MIT) are included unmodified; **Three.js** (MIT) is loaded from a CDN.
- NVIDIA Nemotron and Cosmos models are used through APIs; no model weights are redistributed.

The complete list and license texts are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
