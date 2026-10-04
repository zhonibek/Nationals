# RoboProof prototype

Reproducible adversarial experiments around the **actual Nationals C++ control core**. The CPU baseline executes `simulator/control.wasm`, checks its source/build hashes, and records a complete virtual world for every result. It does not replace the robot library or change production gains.

## Quick start

For the optional **future LocateAnything vision path**, see [perception preparation](perception/README.md). Native output parsing and reviewed calibration-to-task contracts are implemented; runtime, camera capture, real-model evaluation and movement approval wiring remain disabled. The dashboard's Vision panel reports this honestly. It is not a trained movement policy.

For **local Nemotron task coordination connected to the original Simulator and iraLIB**, see [Nemotron setup, UI and limits](NEMOTRON.md). It is separate from the legacy failure investigator and does not replace or train the motion policy. The new RobotAI panel saves its own sessions/results to disk; older scalar dashboard batches remain in memory.

The **Motion lab** now provides canonical controller-only and untrained policy-interface experiments, disk-backed transitions, exact server replay and checked playback inside the original 2D/3D Simulator. `node roboproof/cli.js motion-check` validates the bounded reach environment; `motion-evaluate --suite 2` compares scripted baseline/adapter on 24 frozen worlds. `roboproof.motion_env.MotionEnvironment` gives Python reset/step access to that same Node engine, not a second plant. See [contract](../docs/MOTION_EPISODE_CONTRACT.md). No motion learner/gradients/checkpoints are implemented here yet.

Run from the repository root with Node.js 22 or newer:

```powershell
node roboproof/cli.js nominal
node roboproof/cli.js demo --count 256 --holdout 128 --generations 3 --population 32 --out roboproof/runs/latest
node roboproof/cli.js replay --file roboproof/runs/latest/counterexample.json
node roboproof/server.js
```

Open `http://127.0.0.1:8766`. The explorer loads the latest report, can run new batches and replay selected scenarios. It also accepts an exported report offline. Interactive batches are capped at 1,000; use the CLI or tensor backend for larger experiments. The server binds only to loopback, serves an explicit asset allowlist, rejects foreign origins/Host headers and executes simulations in a bounded worker. Live reports remain in memory; CLI reports are saved.

```powershell
node roboproof/cli.js sample --count 10000 --seed 42 --out roboproof/runs/worlds.json
node roboproof/cli.js run --scenarios roboproof/runs/worlds.json --out roboproof/runs/stress
node roboproof/cli.js benchmark --counts 1,100,1000,10000 --out roboproof/runs/cpu-benchmark.json
node roboproof/ingest.js --repo . --out roboproof/runs/repository.json
```

The scenario file is the editable input: physical parameters, sensor imperfections, start/goal, deadline, thresholds, seed and controller configuration. `scenario-spec.json` declares nominal values, bounds and the sampling distribution. Bounds are **engineering assumptions, not identified properties of a physical robot**. Invalid/out-of-range/nonfinite inputs fail rather than being silently repaired. A batch may not reuse scenario IDs.

## Outputs and verification

`report.json` contains every scenario/result, nominal gate, measured CPU timing, search history, worst-case telemetry and (for `demo`) diagnosis, candidate selection and paired regression counts. `counterexample.json` can be independently replayed; `scenarios.json` preserves the original population. Outputs and local Python environments are gitignored. Reproduction checks the simulator hash, WASM hash, Node version/platform and deterministic numerical outputs. Different backends are compared with explicit tolerance, not claimed bit-identical.

The default demo evaluates three isolated reference-time scales, selects using the training set only, then tests the same selected configuration on new holdout worlds and all adversarial examples. A proposal is accepted only with positive net improvement on training and holdout and no introduced failures. **A rejected proposal is a legitimate result.** The system does not apply a patch to firmware. Extra motion time is visible, and the physical deadline stays fixed. It does not tune its own pass thresholds.

To verify a supplied candidate such as `{"reference_time_scale":1.5}`:

```powershell
node roboproof/cli.js verify --scenarios roboproof/runs/latest/scenarios.json --candidate roboproof/runs/latest/candidate.json --holdout 128 --seed 42 --out roboproof/runs/verification
```

## Physics and measurement contract

The new headless plant is a documented reduced-order model, **not** a full port of the existing browser plant. State contains planar position, heading, field velocity, yaw rate and four wheel rim speeds. The controller receives sensor-derived odometry and measured wheel speed. Truth enters feedback only with the explicitly named `ground-truth-baseline` localization mode.

Motor response is first order with load coupling; contact forces are limited to `friction * mass * 9.81 / 4`. The backward-Euler wheel/contact update uses an assumed effective wheel mass of 0.6 kg and at most 1 ms substeps. Battery voltage clips terminal voltage. Chassis force/torque is integrated with mass/inertia; lateral drag is `mu_lateral*m*g*tanh(v/0.05)`, yaw drag `0.05*omega`. No battery current/sag, collision, full motor electrical circuit, mechanism, safety lease, PROS scheduling or field boundaries are modeled here. Existing browser gameplay stays separate.

Encoders report wheel rim speed; optional IMU reports heading. Each channel supports Gaussian noise, bias, drift, period, sample-and-hold dropout, quantization and latency. Latency/period round **up** to controller ticks; the effective discretization is part of the model. Two scenario-local LCG streams drive Box–Muller noise, independent of batch order. RNG is not used for cryptography. Default localization is encoder-only, reflecting the currently configured robot; IMU settings have no control effect until `encoders-imu` is explicitly selected. The initial physical offset is not disclosed to encoder odometry.

## Metrics and score

Position uses meters; heading uses radians, clockwise from +Y. Spatial path deviation is shortest distance to the straight start/goal segment; tracking error is distance to the time-indexed reference and is reported separately. This first task adapter follows a quintic point-to-point motion; it does not claim whole autonomous-routine or arbitrary path coverage.

All failure categories may coexist. Endpoint/final-heading, maximum spatial-path/localization error, saturation fraction, post-profile heading sign crossings outside a 0.02 rad deadband and physical settling/deadline gates are evaluated against each scenario's saved thresholds. `first_divergence` currently means first path/localization threshold crossing, not every possible failure trigger. Completion requires physical endpoint, heading and velocity dwell after the reference duration. PID terms are not exposed by the existing WASM ABI and are explicitly unavailable.

For finite controller/plant results, the ranking score is:

```text
endpoint / endpoint_threshold
+ max_path / path_threshold
+ final_heading / heading_threshold
+ max_localization / localization_threshold
+ 2 * saturation_fraction
+ oscillations / oscillation_threshold
+ (timeout ? 2 : completion_time / deadline)
```

Invalid control/numerical states score 1,000,000 and fail. The score ranks tests, not certified safety. Monte Carlo pass fraction only describes the sampled distribution. Evolutionary results are intentionally biased and are reported separately, never mixed into a population estimate.

## Evidence and AI

Default diagnosis is deterministic investigation: replay, source inspection, first path/localization crossing, and one-at-a-time sweeps. It reports `llm_status: NOT CONNECTED`, uncalibrated confidence and the actual number of failed sweeps even when the nominal scenario passes. Sensitivity is not causality or a stability-margin measurement. The optional `local_agent.js` bridge also uses fixed rules/templates, not PyTorch or an LLM; `local_model.json` uses portable `node` on PATH. Historical unsupported AI labels remain flagged rather than becoming new inference evidence.

An optional model can investigate using the same bounded tools:

```powershell
node roboproof/agent.js roboproof/runs/latest/counterexample.json model-command.json roboproof/runs/agent-diagnosis.json
```

`model-command.json` contains an executable argv array, e.g. `["python", "my_model_bridge.py"]`. The trusted bridge receives one JSON document on stdin, returns one JSON action on stdout and exits. The request contains `instructions`, the failure summary and the accumulated `history`. The model chooses either a permitted tool or a structured final diagnosis. `agent.js` documents the exact JSON contract; it requires source, telemetry, counterfactual evidence and valid evidence indices before accepting a conclusion. There are at most 20 turns, 36 sweep/replay simulations, 45 seconds per bridge invocation and 256 KiB output per invocation. No shell evaluation is used, and model-generated actions cannot execute commands or edit source. The explicitly configured bridge itself is trusted local software; if it calls a remote provider, source excerpts and telemetry leave the machine through that provider. No provider, credentials or model are bundled; **live LLM investigation remains UNVERIFIED** until a bridge is supplied and exercised.

## Other backends and tests

Deep ML is now a required direction, not just tensorized simulation. [ML readiness](ml/README.md) documents real local neural training, leakage-safe dataset preparation, baseline evaluation, learned experiment selection followed by actual WASM confirmation, and an allowlisted/fail-closed AMD job bundle. These are explicitly legacy-plant experiments: canonical browser/headless integration and real AMD training remain open. No live LLM/provider connection is implied.

See [GPU instructions](gpu/README.md) for the PyTorch tensor translation, CPU/WASM parity and strict ROCm selection. Tensorized execution is not itself evidence of AMD acceleration. See [Pedro differential instructions](pedro_reference/README.md) for executing the unchanged original Java algorithms against the C++ WASM implementation. Missing Java/ROCm support is BLOCKED, not a pass.

```powershell
node --test tests/roboproof.test.js tests/roboproof-server.test.js tests/roboproof-pedro.test.js tests/roboproof-ingest.test.js
node tests/control-core.test.js
```

Architecture, audit evidence, priorities and measured/blocked gates are in `docs/architecture.md`, `docs/progress.md`, `docs/known_issues.md`, `docs/decisions.md`, `docs/benchmarks.md`, `docs/repository_map.json`, `docs/simulator_audit.md` and `docs/pedro_mapping.md`.
