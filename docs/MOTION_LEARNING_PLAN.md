# Integrated robot motion-learning plan

Updated 2026-10-05. This is the implementation checklist for this chat, not a claim of verified movement improvement. The user clarified the primary objective: an AI that improves robot movement through experience, using the **existing Nationals-work3 Simulator and iraLIB/control code**, with VEX gameplay as a later scoring benchmark. This supersedes failure prediction as the primary product direction.

## Mission and product boundary

**Teach the existing robot to move better through simulated experience, and demonstrate improvement on unseen conditions.**

The learning policy selects motion references/actions. The project's existing production controller executes them. The existing simulator produces motion, observations, collisions and, for validated game tasks, scores. Training updates the policy. Independent evaluation determines whether its movement really improved.

Do not build another robot, another physics implementation, an unrelated Gym toy environment or a parallel game. A training wrapper and policy modules are allowed inside this repository, but must call the original Simulator. Browser presentation and headless training must share the exact same engine and action interface.

“iraLIB” here means the user's existing robotics/control stack in this project, not a new replacement library. The source names currently include `nationals`, LemLib and other modules. The audited active X-drive control chain is trajectory -> LTV-LQR -> wheel kinematics -> wheel PI/feedforward -> motor voltages. Merely having other PID/LQR/follower files does not mean they are active, interchangeable or supported training actions.

## Current state: what can be reused

Local high-level coordination was added on 2026-10-01 at the user's request: official quantized Nemotron -> validated reach task -> separate user approval -> original Simulator/iraLIB execution, exact replay and saved evidence; the existing visual Simulator loads that same task. Live local inference was demonstrated. This auxiliary milestone does not complete action adaptation, motion-policy training or the learned-improvement gate. See [Nemotron integration](../roboproof/NEMOTRON.md).

| Existing component | Role in the integrated implementation |
| --- | --- |
| `simulator/engine.js` | Original `VexRobotSimulator`, `update(dt)`, `stepHolonomicPhysics`, odometry and queued motion; shared browser/headless motion/plant authority |
| `simulator/simulator.js`, `motion.js`, `headless.js` | Original renderer/planner/UI, versioned reach-benchmark adapter and source-checked headless factory; no separate physics |
| `simulator/override.js`, `override-geometry.js`, `override-dynamics.js` | Existing game state, field/object geometry, interactions, collisions and score calculation; preserve and test rather than clone |
| `simulator/override-autonomy.js` | Existing fleet/autonomous action execution and scripted game baseline |
| `simulator/index.html`, `simulator.css`, `override-view.js` | Main human-visible simulator and training/evaluation playback destination |
| `simulator/control-runtime.js`, `control.wasm`, `control-build.json` | Source-checked C++ controller/reference adapter; independent WASM state per robot/environment |
| `include/subsystems/control/Cascade.hpp`, `include/autonomous.hpp`, `src/subsystems/control/HolonomicMotion.cpp` | Existing control semantics and physical-runtime API; do not silently replace or rewrite them |
| `config/robot.json`, `robot-config.js`, config generation tooling | One robot geometry/configuration source; do not maintain an ML-only robot configuration |
| `roboproof/core.js`, analysis, CLI, server, telemetry and regression tools | Reuse provenance, experiment records and comparisons after adapting them to canonical Simulator contracts |
| `roboproof/ml/` | Reuse checkpoint/budget/evaluation infrastructure where appropriate; the current failure predictor is auxiliary, not the motion policy |
| `roboproof/gpu/` | Existing legacy tensor translation is comparison evidence only; it is not already a GPU port of the original simulator |
| `tests/`, `simulator/test_*.js` | Preserve original control, geometry, dynamics, match, configuration and RoboProof regression coverage |

The current neural failure predictor and reduced `roboproof/sim.js` are historical/auxiliary experiments. Preserve their evidence and label them legacy. Do not treat their 128-scenario training, animation or tensor benchmarks as learned movement, canonical simulator integration or physical validation.

Existing Node tests load the actual `VexRobotSimulator` in a VM; this demonstrates that headless access is possible. It is a useful migration starting point, not the desired production architecture of a fake DOM plus a duplicate physics loop.

## Target execution loop

```text
Existing Simulator episode (same robot configuration and game/field)
    -> sensor-derived observation
    -> learned motion policy
    -> validated, bounded motion action/reference adapter
    -> existing source-checked iraLIB/C++ controller
    -> existing Simulator physical/game step
    -> next observation + measured reward + termination
    -> policy update during training
```

Rendering reads state and never determines training outcomes. The policy's observation must not secretly include physical truth; truth is available to the evaluator and renderer. If an initial experiment intentionally uses privileged observations, name that separate mode and make no sensor-realistic claim.

## Implementation phases and gates

### 1. Freeze task, interfaces and ownership — benchmark foundation implemented

- [x] Trace the original simulator's update order, motor model, estimator, action execution, game/fleet dispatch, reset and terminal behavior. Record units and heading conversions at the C++ boundary.
- [x] Define a versioned benchmark contract for reset(seed, configuration, task), observe(), applyAction(action), step(fixedDt), termination, reward components and telemetry. Continuous policy-action and complete transition-log contracts remain phase 3 work.
- [x] Scope the first task to **reach a target pose reliably and efficiently inside the existing simulator**. Preserve the game environment, but do not require an unfinished whole-match strategy for the first learning smoke test.
- [x] Specify supported randomization from parameters the original plant actually exposes; reject unsupported faults/options rather than pretending the old reduced-model schema maps automatically.
- [x] Keep the existing scripted/controller-only robot as the baseline. Define evaluation worlds and improvement metrics before training.

Implemented evidence, units, supported parameters and limitations: [Original-Simulator episode contract](MOTION_EPISODE_CONTRACT.md). Scripted baseline succeeds in 12/12 frozen evaluation worlds; this is not learned improvement.

**Gate:** one documented, testable task/action/observation contract with the exact original Simulator and controller identified. No new policy architecture is chosen to compensate for an undefined environment.

### 2. Make the existing Simulator a shared deterministic engine

- [x] Extract or wrap the original `VexRobotSimulator` and supporting game modules so headless execution does not require a DOM, canvas, Three.js or browser timers.
- [x] Separate rendering, UI presentation, rumble/timers and keyboard bindings from authoritative physical/game state. Preserve original plant/game constants; bounded display arrays remain derived engine data. Rumble now follows simulation time, including pause.
- [x] Use that shared engine in `simulator/index.html` and the new RoboProof motion/headless runner. Do not copy `stepHolonomicPhysics` into a new backend or substitute `roboproof/sim.js`. Legacy failure-testing commands retain their explicitly separate plant.
- [x] Reset all robot/controller integrators, odometry, wheel/motor state, game objects, fleet entries, clocks, queues, reward history and RNG state at episode boundaries. Give each environment its own WASM instance and RNG streams.
- [x] Freeze physical dt and keep presentation reads independent. Rendering reads, interleaving and another environment's actions do not change the benchmark trace; configurable continuous policy cadence remains phase 3 work.
- [x] Test fixed-seed browser-realm/headless same-action parity through the same entry points; test reset isolation and interleaved environments. Keep existing geometry/dynamics/match tests passing. Captured original traces remain identical.

**Gate:** the original browser and headless runner execute the same supported world and action log with matching state, score/events and metrics in the same runtime. Cross-runtime tolerances are explicit. This proves software consistency, not real-world fidelity.

### 3. Add the motion-policy environment inside this project — implemented, training separate

- [x] Implement bounded original-engine JSON-lines access and a standard-library Python adapter with fixed 4-action/34-observation arrays, versioning, timeouts and owned process teardown. No Python physics or external toy environment.
- [x] Initial observations: sensor-derived pose/velocity, target-relative error, wheel measurements, previous action and remaining episode time. Game/object sensing is not part of reach-benchmark v1.
- [x] Freeze the initial action as bounded pace and body-relative residual references around the production profile. Rate/acceleration limits and field bounds are tested; the C++ controller resets once on activation, not per policy tick. This is reference adaptation, not a general path planner or direct motor learning.
- [x] Maintain fixed 100 Hz control/physics and configurable policy cadence (default 20 Hz). Test malformed/nonfinite actions, stale-action timeout, stop, bounded references and explicit fault observation fallback.
- [x] Reach-task benchmark reward: measured reduction in target distance plus terminal pose success, with documented time/effort/contact penalties. Calculate every component from authoritative simulation state, not an animation, a predicted outcome or a target marker.
- [x] Define success using position, wrapped heading and physical settling, with fixed deadlines. Distinguish true terminal states from time-limit truncation.
- [x] Record episode seed/configuration and identities, pre-action observations, applied actions, actual reward components, terminal flags and evaluator-only physical traces. Exact server replay and checked original 2D/3D recomputation are implemented; neither calls a model/policy during playback.

**Gate:** scripted and random-action agents can complete bounded episodes through the real shared Simulator; their observation/action/reward traces replay correctly. No learned improvement is claimed yet.

Evidence: `motion-check` passes and retains source/runtime-bound readiness plus full fixture reports under ignored `roboproof/runs/motion/`. Python/API/store tests pass. Expanded frozen v2 baseline is 6/24 and nominal adapter 7/24; these negative stress results remain visible and are not motion-learning evidence.

### 4. Validate the task and VEX scoring environment

- [x] Preserve and test original reach-task stepping, field contact, reset, true settling, deadline truncation, sensor/truth separation and fixed-time browser/headless parity. Physics parameters remain approximate; no physical calibration or full-game certification is implied.
- [x] Test baseline/reference adaptation and failed reach cases, including battery resistance/friction/load and heading wrap. Freeze 24 harder worlds with explicit acceptance metrics and training exclusion. Do not tune on evaluation cases or hide deadline-pressure failures.
- [ ] For the later game task, audit the existing score/interactions against a pinned game-rule version; record all approximations and unsupported mechanisms. Use current official rules when doing that audit rather than assuming this plan verifies them.
- [ ] Verify acquisition/release, collisions, object ownership, score timing, match phases and score finalization. Resolve score/reward exploits before calling a policy's higher score an improvement.
- [ ] Preserve a simple scripted VEX routine using the existing autonomy/fleet implementation as the scoring baseline.

**Gate:** reach-task mechanics are testable; game-score learning stays disabled until the necessary game/scoring rules pass their own audit. A perfect digital twin is not required for a software demo, but its approximations must be visible.

### 5. Train the first genuine movement policy locally

- [x] Implement a compact PPO Beta actor-critic using the existing pinned PyTorch 2.8 CPU runtime, with two 32-neuron hidden layers. Test bounded actions, reset behavior and terminal/truncation bootstrap; reuse the existing environment rather than new physics.
- [x] Execute real CPU gradients with logged rewards and changed actor weights. Three seeds completed eight updates each; this is a pipeline smoke, not evidence of better movement.
- [x] Add curriculum: simple target movement, then heading/starting-position variation, then supported physical variation. Frozen evaluation worlds remain excluded from curriculum and tuning.
- [x] Save actor/critic, optimizer, fixed engineering-unit normalization, both RNG states, configuration, source/engine hashes and episode/step counts. Exact interrupted/resumed exported actor equality is tested at committed complete-rollout/episode boundaries; no arbitrary mid-episode WASM restoration.
- [x] Bound environment steps, collection time, CPU worker count and checkpoint count/read sizes. Show committed history in the existing Motion lab and learned experiments in the original Simulator. CLI training is explicit; browser refresh only reads artifacts.
- [x] Keep failure prediction separate and unused in these rollouts. Rewards and transitions come from the canonical Simulator, never predicted movement.

**Gate:** a real motion policy trains through the original Simulator and existing controller; interrupted runs resume consistently at the declared boundary. Learning quality is a separate next gate.

Phase 5 smoke delivered 2026-10-05: run `d2f440b7-5b37-4f65-bc15-304d203527a0`, seeds 42/43/44, 3,658 policy steps across 76 episodes, 24 total PPO updates; all actors changed. [Setup, provenance, budgets and limitations](../roboproof/motion_learning/README.md).

### 6. Prove improvement and show it in the original Simulator

- [x] Freeze policies before evaluation. Compare initial untrained policy, learned policy and existing scripted/controller-only baseline on identical frozen worlds, deadlines, actuator limits and task definitions.
- [x] Retain success rate, endpoint/heading error, time, contact duration and effort proxy for every world, including failures. Saved explicit baseline/learned episodes replay the real trajectories; calibrated energy and full-game score remain separate.
- [x] Evaluate the frozen new start/goal/physical-condition corpus across three independent training seeds and retain variation; no best-seed selection using the corpus.
- [ ] Require a measured improvement against a predeclared baseline/metric without unacceptable regressions in the other metrics. If this fails, retain the negative result and revise the hypothesis—not the scoreboard.
- [x] Add saved learned-policy experiments, frozen comparison, history refresh and gated verified-policy selection to the existing Motion lab. Exact action playback uses the original 2D/3D Simulator, not substitute animation/physics.
- [ ] For game learning, distinguish optimizing a simulated scoring strategy from robust movement and from physical robot transfer.

**Gate:** reproducible before/after movement improvement on unseen worlds is visible in the existing Simulator. A successful gradient update, larger network or higher training reward alone does not pass.

**Not passed:** controller, untrained actor and all three learned actors each succeed in 6/24 frozen worlds. Baseline remains the default and verified-policy selection stays disabled. Preserve this negative result. Develop on training/development-only cases and reserve a new untouched final corpus before further held-out improvement claims; never tune against these inspected failures.

### 7. Scale with AMD only after local correctness

- [ ] Await the supplied API/provider documentation, authorized access, GPU models/count/VRAM, OS/HIP runtime, CPU allocation, storage and explicit cost/time limits. No secret in source/chat, guessed endpoint or unapproved paid compute.
- [ ] Probe capabilities read-only, then run a small actual policy training job with strict ROCm selection and retained metadata. A fake provider or CPU fallback is not AMD execution.
- [ ] Package the canonical Simulator/controller/WASM/configuration and policy code together, with hashes and required dependencies. The current ML-only legacy job bundle is insufficient for movement training.
- [ ] Initially parallelize independent original-Simulator workers on CPU and train the neural policy on the AMD GPU. GPU policy training does not mean the JavaScript simulator itself runs on the GPU.
- [ ] Add any accelerated plant/game backend only after it matches the original Simulator contract and passes state/metric/event parity. Do not abandon the shared-engine requirement for throughput.
- [ ] Measure actual environment throughput, policy-update time, total training wall time and resource/cost use; implement verified cancellation, artifact retrieval and bounded retries under the supplied provider interface.

**Gate:** genuine bounded AMD policy training on transitions from the canonical project environment, not a legacy failure-predictor benchmark mislabeled as movement learning.

### 8. Deliver the integrated demonstration; physical transfer is separate

- [ ] Demonstrate one robot/task: controller-only baseline -> practice/training -> frozen learned policy -> unseen-world comparison -> before/after original-Simulator playback.
- [ ] Expand to validated VEX scoring only after movement learning and game-rule gates hold. More score is better for that task, but score exploits, collisions and regressions remain explicit.
- [ ] Retain a runnable setup, configuration, source identities, saved policy/checkpoint, evaluation corpus, machine-readable results and limitations.
- [ ] Keep physical deployment disabled by default. Hardware transfer requires calibration, sensor-realistic inputs, compatible control timing, bounded supervised tests and an independent safety review. Simulated score is not physical performance evidence.
- [ ] GitHub publication is a separate release action: inspect secrets/generated artifacts and preserve the existing unrelated Pedro deletions. No automatic firmware merge or force push.

**Gate:** the product demonstrates learned movement—not merely testing, diagnosis, a prerecorded animation or an LLM narration.

## Operating rules for the agent

1. Work in `Nationals-work3`. Extend existing Simulator and iraLIB/control interfaces; keep a single authoritative robot/physics/game model.
2. Follow phases 1 -> 2 -> 3 -> 4 (reach-task gate) -> 5 -> 6 -> 7 -> 8. Game-rule work can proceed independently, but no game-score claim precedes its gate. AMD access research can proceed early; expensive training cannot bypass local correctness.
3. For each phase, record exact changes, focused tests, evidence and remaining limitations in progress documentation. Mark a gate complete only with actual retained evidence.
4. Preserve established behavior and user changes. Do not reorganize unrelated source, restore/delete the reference tree, switch production controllers or alter physical firmware gains to manufacture improvement.
5. Reuse useful verification infrastructure, but do not treat historical failure-prediction results as motion-policy evidence. Existing completed tests remain historical evidence for their own scope.
6. No live cloud inference is required for motion-policy training. Local Nemotron now coordinates supported tasks; Nebius/cloud execution and explanations remain optional future work, not the movement learner or its reward oracle.
7. No training, engine refactor or paid/remote execution is performed by this planning edit itself.

## Immediate next deliverable

The original engine, persistent-reference interface, real PPO CPU gradients, checkpoint/resume, JSON neural inference and original-Simulator playback are delivered. First independent frozen evaluation is negative, so the next deliverable is measured generalization improvement: diagnose learning limitations using training/development-only worlds, train a longer predeclared bounded experiment and evaluate a separately untouched final corpus. Obstacle perception/planning, game learning and hardware calibration still need their own contracts and evidence. Nemotron, LocateAnything and AMD are auxiliary/later gates, not substitutes for movement learning.

## Optional perception track — preparation added 2026-10-02

At the user's request, LocateAnything is prepared for future visual target grounding, not enabled as a movement policy or used to bypass the main training phases. See [perception contract](../roboproof/perception/README.md).

- [x] Define versioned image/frame/model-output provenance and parse the model's native normalized boxes/points, with strict malformed/truncated-output rejection and no invented confidence.
- [x] Implement an offline reviewed floor-point + fixed-plane calibration gate producing the shared original-Simulator reach task. Bind image/pose/profile identity, freshness, expiry, held-out measurement declarations, coverage and field bounds; prohibit automatic box-center navigation.
- [x] Expose a read-only preparation status and dashboard section. Keep model runtime, capture, cloud calls and execution endpoints disabled; provide an explicitly synthetic fixture, not fake live inference.
- [ ] Review intended-use/license constraints, custom model code, pinned weights and compatible deployment resources before model setup.
- [ ] Build a bounded inference worker and independently verified image/calibration provenance; demonstrate real grounding on held-out images and explicit latency/ambiguity handling.
- [ ] Add reviewed/revalidated task approval and original-Simulator playback with reachability/obstacle gates. Camera-click navigation and physical deployment remain unavailable.

This optional track does not block coordinate-based motion learning. LocateAnything supplies visual target evidence, Nemotron coordinates supported tasks, the future policy learns movement, and iraLIB executes bounded control in the original Simulator.

## Agent Skills coordination track — 2026-10-06

The existing local Nemotron coordinator now loads four reviewed workflows: inspect robot configuration, prepare a reach-pose experiment, diagnose one explicitly selected saved experiment and review saved PPO/frozen-evaluation evidence. [Skill contracts and usage](../roboproof/skills/README.md) describe the bounded native tools, on-demand instructions, permission checks and saved hashes. This extends the full project; it does not introduce an independent simulator, robot library or cloud agent.

Skills supply procedural context, not weight training. Movement proposals still require separate approval and execute through original Simulator/iraLIB. Diagnosis and learning review are read-only, keep measured facts separate from model commentary and do not tune, retrain, rerun evaluation or promote a policy. The negative 6/24 improvement result remains unchanged.

Next coordination gate: predeclare multilingual workflow-selection and evidence-grounding cases, measure failures/latency, and retain ambiguity/injection failures. Deeper time-series tools need their own measured tests. This track cannot replace meaningful AMD execution, learned movement generalization, physical calibration, full-game mechanics or camera calibration.
