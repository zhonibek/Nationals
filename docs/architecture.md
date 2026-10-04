# RoboProof architecture

Updated 2026-10-02. This prototype extends Nationals' shared original Simulator; physical robot code and user-deleted Pedro worktree files are not reorganized or restored.

## Boundaries and ownership

| Component | Responsibility | Boundary |
| --- | --- | --- |
| `roboproof/ingest.js` | Supported repository evidence extraction | File evidence and hashes; unknown frameworks stay unknown |
| `roboproof/core.js`, `scenario-spec.json` | Versioned full worlds, seeded distributions, validation, provenance | Explicit SI configuration and software identity |
| `roboproof/sim.js` | Scalar truth physics, sensors, localization, metrics | Actual source-checked C++ WASM controller receives estimates only |
| `simulator/control-runtime.js`, `control.wasm` | Existing Nationals production control adapter | `reference[6]`, `feedback[10]`, explicit dt -> four volts/targets |
| `roboproof/gpu/` | PyTorch translation and scenario-axis batching | CPU parity required; ROCm hardware must be detected, never inferred |
| `roboproof/ml/` | Versioned data, grouped splits, deep training, acquisition, confirmation and job packaging | Legacy-only opt-in; final holdout not evaluated during training; no live remote provider |
| `roboproof/analysis.js` | Monte Carlo summaries, evolutionary hunt, tools, paired regression | Test world/seed/thresholds fixed across candidate comparisons |
| `roboproof/agent.js` | Optional external-model tool loop | Whitelisted evidence tools; no model-driven shell or source writes |
| `roboproof/nemotron.js`, `nemotron-store.js`, `nemotron-worker.js` | Local Nemotron task coordinator, durable sessions and approved reach execution | Native allowlisted tools; no execution before approval; canonical original engine and actual controller; not motion learning |
| `roboproof/perception/`, `tools/prepare-perception.js` | Future LocateAnything output parsing and reviewed fixed-plane calibration-to-task proposals | Offline/imported evidence explicitly unverified; shared task schema; no model runtime, camera or execution endpoint |
| `roboproof/cli.js` | End-to-end artifacts, candidate selection, replay, timing | Headless and reproducible; no automatic production merge |
| `roboproof/server.js`, `worker.js` | Local interactive simulation service | Loopback, same-origin, bounded worker and requests |
| `roboproof/dashboard/` | Trajectory/metric/parameter/diagnosis exploration | Real report data; absent evidence stays absent |
| `roboproof/pedro_reference/` | Original Java vs compiled C++ geometry probes | Original Java is read-only; availability and mismatches explicit |

Orchestration owns architecture, progress, decisions, issues and benchmarks. Audit, Pedro, GPU, ingestion and dashboard work have disjoint file ownership. No subsystem silently changes another's model or claims another's quality gate.

## Pipeline

The optional future vision branch is image + description -> LocateAnything -> pixel geometry -> reviewed floor-plane selection/calibration -> the same motion-task schema -> future separate approval. Only the parser and offline proposal gate are implemented; `GET /api/perception/status` and the dashboard show preparation, without inference. This adds no second physics model and does not finish the movement learner. See [perception contract and readiness gates](../roboproof/perception/README.md).

RobotAI's local task pipeline is separate from the legacy failure-testing sequence: user task -> real local Nemotron native tool call -> shared task-schema validation -> persisted proposal -> explicit approval -> original Simulator/iraLIB worker -> measured report + exact replay. `simulator/nemotron-task.js` opens that saved task in the existing visual UI; it does not invent another plant. Only the task and supported contract are sent to the loopback model; cloud endpoints are disabled. See [model integration](../roboproof/NEMOTRON.md).

1. Inspect supported repository and verify the existing controller build manifest.
2. Define nominal/start/goal and the complete physical/sensor uncertainty distribution.
3. Execute deterministic scalar closed-loop runs and retain truth separately from estimates.
4. Rank failures and mutate elite worlds strictly within saved parameter bounds.
5. Export the worst counterexample and replay it independently.
6. Investigate source, first divergence and parameter counterfactuals through explicit tools.
7. Evaluate isolated reference-time configuration candidates on training worlds.
8. Test the selected candidate on identical training worlds, unseen holdout worlds and discovered counterexamples; count fixed failures and new regressions independently.
9. Explore the evidence and export the proposal. Firmware application remains a user decision.

## Fidelity contract

As of 2026-10-01, the motion-learning foundation uses `simulator/engine.js`, extracted from the original browser Simulator, with `simulator/motion.js` and `simulator/headless.js`. Browser rendering and RoboProof's `motion-baseline`/`motion-replay` commands share this original plant and production controller. Seeded reach episodes, evaluator-only truth, sensor-only observations and measured reward are documented in `MOTION_EPISODE_CONTRACT.md`. This is a scripted benchmark, not trained motion learning.

The older failure-testing path (`roboproof/sim.js`) still source-shares only the active C++ cascade. Its headless plant is a separate reduced-order model with first-order motors, wheel/contact coupling, traction limits, voltage limits, inertia, drag, noisy/delayed sensors and encoder localization. It deliberately does not claim browser-plant, PROS runtime, safety watchdog, manipulator or real-robot equivalence. The historical `simulator_audit.md` maps those earlier boundaries. This old path does not become the motion-training plant by sharing a controller.

The three dimensions of evidence are separate: deterministic replay, cross-language numerical agreement and physical predictive validity. Passing the first two cannot establish the third. Model constants/distributions remain uncalibrated.

## Extension points

The required deep-ML path is **actual controller/simulation results -> audited labels -> grouped dataset -> trained failure predictor -> new experiment proposals -> actual simulator confirmation**. It never feeds predicted labels into verification or changes production gains. Neural gradient training and LLM investigation are distinct capabilities. `roboproof/ml/README.md` specifies the shipped local slice and remaining canonical/AMD/evaluation gates; the existing reduced plant cannot be relabeled canonical merely because a neural model trained on it.

Additional robot/controller adapters should preserve the explicit reference/feedback/voltage boundary and supply their own schema validation and parity tests. Additional tasks need their own geometry metric, reference generator and completion logic. GPU engines must compare identical scenarios against the scalar oracle before benchmarks are treated as verification results. No current claim of arbitrary robotics repository ingestion, ROS2 or universal safety is made.
