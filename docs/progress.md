# RoboProof progress

## Primary direction clarified — 2026-10-01

The user wants **learning robot movement through experience**, using the existing Simulator and iraLIB/control stack, with VEX scoring as a later validated benchmark. Failure prediction is no longer the primary product. The execution checklist is [Integrated motion-learning plan](MOTION_LEARNING_PLAN.md). Its first dependency is a shared original-Simulator engine and explicit episode/action/observation contracts; no separate physics/game project is authorized. Motion-policy training and measured movement improvement are **NOT IMPLEMENTED** by the planning update. The earlier evidence below remains valid only for its original failure-testing/legacy-ML scope.

Updated 2026-10-02. Master-plan success is **not yet complete**. This file distinguishes shipped local functionality from measured gates and external requirements. Earlier failure-testing/ML evidence retains its original date and scope.

## Implemented

- Optional LocateAnything preparation (2026-10-02): bounded native box/point parser, frame/selection provenance checks, reviewed fixed-plane calibration gate, offline proposal CLI and shared original-Simulator task schema. Read-only API/dashboard readiness; runtime, capture, automatic execution and cloud disabled. No weights downloaded or real LocateAnything inference performed. Synthetic fixtures are labeled and are not model accuracy or physical calibration evidence. See [perception preparation](../roboproof/perception/README.md).

  Validation: 17 new perception tests pass, including explicit synthetic-task execution in the original source-checked Simulator and exact replay. Focused perception/Nemotron/server/original-engine suite: 43/43 pass. Full `node --test tests/*.test.js simulator/test_*.js`: 102 pass, zero failures, one preexisting external-Java compiler/runtime skip (67.997 seconds). Generated configuration check passes. The in-app browser renders the preparation-only Vision panel without horizontal overflow or console warnings/errors; the prior disk-backed Nemotron result still restores successfully. This is contract/UI regression evidence, not live perception or training evidence. Original controller/physics files and all 91 user-deleted Pedro files remain untouched by this update.

- RobotAI motion foundation (2026-10-01): extracted original shared Simulator engine, preserved original physics/game trace digests, DOM-free source-checked headless controller, versioned reach benchmark with sensor-only observations and measured success/reward, isolated seeded resets, original-engine RoboProof baseline/replay commands and 12 frozen evaluation worlds. Scripted baseline: 12/12 success. Motion-policy training is still not implemented. See [episode contract](MOTION_EPISODE_CONTRACT.md).

- Phase A: current repository/control/sensor audit, machine-readable source map, Pedro algorithm mapping and original-simulator limitations.
- Phases B–D: source-checked real C++ controller execution, full validated seeded scenarios, reproducible sensor/physics loop and telemetry with separate truth/estimate.
- Phase E/search: scalar batch CLI, Monte Carlo, bounded elitist failure search, counterexample export/replay and measured timing.
- Verification: isolated candidate configuration search, matched worlds, held-out worlds, failure-fixed/regression counts and explicit rejection.
- Investigation: deterministic source/telemetry/counterfactual tools; optional external-model protocol with evidence validation and execution budgets.
- Exploration: local worker API and data-driven report viewer; CPU/Torch/Pedro integration status is recorded by their executable tests and benchmark outputs.
- Deep ML preparation: versioned pre-run features/observed labels, integrity-checked grouped splits and train-only preprocessing, actual multi-task neural gradient training, majority/logistic baselines, calibration, resumable checkpoints, explicit final-holdout consumption, learned candidate selection and actual controller confirmation. All current artifacts are experimental legacy-plant evidence, not canonical/physical validity.
- AMD readiness: allowlisted dataset/source bundles, strict HIP selection, zero-spend/no-retry configuration, and fake lifecycle/budget/artifact tests. No live provider transport, remote job or AMD training is claimed.

## Recorded experiment

`roboproof/runs/latest/report.json`, seed 42, 256 baseline worlds plus 96 adversarial children and 128 held-out worlds:

- Nominal: PASS.
- Baseline: 30/256 passed (11.71875% of the configured broad stress distribution).
- Selected training candidate: reference time scale 1.5; 36/256 passed, 14 failures fixed, 8 new regressions.
- Holdout: 22/128 -> 24/128 passed, 4 fixed, 2 regressions.
- Decision: REJECTED; no production patch applied.
- Worst exported scenario `evo-2712847358-3-2` replayed with exact recorded output equality in the same runtime.

These are software-model measurements, not physical reliability estimates. The distribution deliberately includes simultaneous physical, calibration and sensing uncertainties. AMD throughput and live LLM investigation are not demonstrated by these numbers.

## Gates

| Gate | Current evidence |
| --- | --- |
| A build | Existing WASM source and binary hashes pass. No new ARM/PROS firmware build claimed. |
| B units | 51/51 RoboProof Node tests and 12/12 Python ML tests pass with no skips (2026-09-30). Prior Python tensor suite: 10 pass, 2 genuine-ROCm tests explicitly skipped. Actual WASM controller parity: 510 lane-steps, max absolute error 1.42e-14; full simulation parity: 2,000 lane-steps, max error 2.60e-13. |
| C Java differential | PASS: 2,077/2,077 cubic Bezier cases against unchanged original Java using Temurin JDK 8u504. 177 intentional low-speed curvature cutoff differences remain recorded; no global follower-equivalence assertion. |
| D nominal | PASS in reduced-order model with actual C++ controller. |
| E stress | 256-world run, 96 adversarial cases; 10,000 CPU scenarios measured in 231.564505 s. Genuine AMD gate remains blocked on compatible hardware/runtime. |
| F regression | Completed training and holdout; proposed change rejected for new failures. |
| G physical | NOT PERFORMED — REQUIRES PHYSICAL TEST. |

## Next highest-value work

Local Nemotron coordination is now a delivered auxiliary milestone: see [integration and live evidence](../roboproof/NEMOTRON.md). Its task proposals and approved canonical-simulator results persist on disk. Two real local model requests proposed distinct 24-inch goals; both approved original-engine runs passed in 2.91 simulated seconds with exact replay. Full Node/control/Simulator regression: 85 passed, zero failed, one external-Java test skipped for unavailable compiler; generated config check passed. The motion-learning priorities below remain unchanged; no cloud credits, policy training or physical deployment were used.

1. Delivered 2026-10-02: persistent bounded reference actions, policy vectors, full transition logs, Python access, disk-backed canonical experiments and checked original-Simulator replay. No independent plant.
2. Validate reach mechanics, then run a real CPU motion-policy training smoke, checkpoint/resume and frozen before/after evaluation. Existing failure-predictor training is not motion learning.
3. Audit game mechanics/scoring before using VEX score as a learning reward. Physical transfer requires separate supervised testing.
4. Scale the verified motion pipeline on AMD only after server documentation/access and explicit spending limits arrive. Optional Nebius/failure prediction remain auxiliary; no remote provider has been used. A real local pretrained Nemotron coordinator is connected, but it is not the motion learner.
5. Identify plant parameters and safely reproduce on physical hardware only with supervised authorization; complete the integrated release gates before declaring the master plan finished.

## ML readiness verification

On 2026-09-30, the complete RoboProof Node suite passed 51/51 tests without skips, including the actual Java differential run using the existing portable JDK. The Python ML suite passed 12/12 without skips. Python tests exercise real gradients, deterministic same-runtime repeated training, interrupted checkpoint resume and fake AMD lifecycle limits. Fake provider tests do not count as remote execution.

A fresh 128-world corpus (seed 5053102) produced 19 passes/109 failures. The deep MLP (3 x 128 hidden units) trained on 78 rows for 20 epochs on CPU; 19 validation, 12 calibration and 19 untouched final rows remained separate. From 256 new candidate worlds, acquisition selected 8 (2 random exploration, 6 model-ranked), and the actual source-checked WASM simulation confirmed failures in all 8. This small biased cohort is **not** proof of advantage over random/evolutionary search. `roboproof/runs/ml-demo/` retains the report, dataset, package, model/card/checkpoint, proposed selections, confirmations and verified offline AMD bundle. No remote GPU job, final evaluation or paid service ran.

Browser checks confirmed the fresh ML report shows model/device/epochs and distinct predicted/measured categories; scenario selection updates the corresponding predictions. Replaying selected `mc-5053103-60` produced exact recorded metric equality and 1,000 actual telemetry samples. The user's `roboproof-scenario (2).json` imported as `mc-42-121`, PASS, endpoint error 0.050366 m and completion 5.62 s; Play advanced through real samples and End stopped at sample 1000. The ML panel hides when a non-ML scenario is loaded. No browser errors/warnings were captured; the new hash text wraps in the narrow panel.

Generated artifacts are ignored by Git; current source hashes must match before model reuse. See `roboproof/ml/README.md` for commands and unresolved gates.

## Validation environment

Java was provisioned only as a portable, SHA-256-verified official Temurin archive under the ignored `.cache/roboproof-jdk` directory; system Java and environment settings were not changed. The retained Java report is `../outputs/roboproof-pedro-20260929-jdk/report.json`; its versionable summary is `roboproof/pedro_reference/validation/summary.json`. To include Java in the Node tests on this machine, set session-local `PEDRO_JAVA` and `PEDRO_JAVAC` to the executables under `.cache/roboproof-jdk/temurin8-504/jdk8u504-b01/bin`.

The repeated Python suite completed in 64.465 seconds using PyTorch 2.8.0+cpu; this is test wall time, not simulation throughput. Original control/configuration and browser-plant tests also passed. No robot firmware, original Pedro source, or existing user deletions were changed.

## Canonical motion-training preparation — 2026-10-02

- Corrected deterministic diagnosis labels and causal/confidence overclaims. A nominal pass with a failed friction sweep now discloses that failure; no PyTorch/LLM inference is claimed. Historical reports are explicitly warned about in the UI.
- Added shared `simulator/policy.js`: four bounded pace/right/forward/heading residual actions, 34 sensor/reference observations, slew/acceleration/field limits, configurable policy cadence and stale-action stop. Production controller state persists between decisions; the active C++ firmware/gains/WASM were not changed.
- Versioned canonical episodes to schema 2 with complete pre-action/action/reward/next-observation/terminal/evaluator traces. Exact same-runtime replay and 1e-9-tolerance browser recomputation use recorded actions, never invented motion or model narration.
- Added a bounded shell-free local Node protocol and standard-library Python adapter, with explicit ownership/close, response identity/timeout checks, reset after terminal/truncation and safe invalid-action stop. Python performs no physics.
- Added a Motion lab using the existing dashboard/server and original 2D/3D Simulator. Canonical experiments save/restore on disk; legacy scalar experiments retain an explicit separate-plant label. No training/inference/paid or physical operations run automatically.
- Froze a harder 24-world evaluation, eight task groups, acceptance metrics and training exclusion. Fresh controller baseline: 6/24 success; nominal scripted adapter: 7/24. Mean elapsed times across all worlds: 7.64625 s versus 7.6454167 s. These are not learned improvement; failures and deadline-pressure cases remain visible.
- `motion-check` passes and retains runtime/source-bound evidence plus complete fixture reports under `roboproof/runs/motion/`. This permits a bounded CPU reach experiment, not full-game, physical, camera or AMD claims.
- Added Windows/Linux CI for Node and Python contracts with pinned CPU Torch; GitHub execution remains unverified until publication. No source was committed/pushed and the user's Pedro deletions remain untouched.

Next: real compact CPU motion-policy gradients, bounded budgets and episode-boundary checkpoints/resume; then frozen policies evaluated across independent seeds. Auxiliary failure-predictor checkpoints are not motion-policy progress.
