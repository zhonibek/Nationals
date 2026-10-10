# Integrated robot motion-learning plan

Updated 2026-10-09. This is the implementation checklist for this chat, not a claim of verified movement improvement. The user clarified the primary objective: an AI that improves robot movement through experience, using the **existing Nationals-work3 Simulator and iraLIB/control code**, with VEX gameplay as a later scoring benchmark. This supersedes failure prediction as the primary product direction.

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

**Not passed:** the historical short smoke experiment remains 6/24 for controller, untrained and learned actors. A longer 64-update three-seed experiment subsequently failed a separately committed 128-world final: controller 87/128, neutral 88/128, untrained 80/128, learned seeds 201/202/203 at 75/83/81 with 12/5/8 controller-success regressions. Effort-proxy savings do not satisfy the success gate. Both negative results remain retained; neither consumed corpus may be reused as fresh evidence. Baseline remains the default and verified-policy selection stays disabled. See [progress](progress.md) for exact run identities, development experiments and limitations.

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

The original engine, persistent-reference interface, real PPO CPU gradients, checkpoint/resume, JSON neural inference and original-Simulator playback are delivered. Longer training and a fresh final have been attempted, but improvement is still unverified. Complete-episode credit and margin-aware demonstration selection each passed a single development seed, but neither reproduced across three fresh seeds. Latest margin-aware validation reached 26/27/25 versus controller 27/32, with 2/0/2 regressions; all strict gates failed. Neutral-pace residual-only diagnostics recovered 28/32 across those actors, but lacked the required 5% efficiency gain and are post-hoc scripted policies, not learned fixes. No favorable seed, checkpoint or oracle is promoted or sent alone to a final.

Public reference timing is now implemented as optional `deadline-context-v1`:145 engineered inputs, required original-planner reset context, unchanged raw34, non-mutating bootstrap, exact resumption and Python/JavaScript parity. However, fresh seeds2001/2002/2003 reached27/26/27 with0/2/1 regressions; all strict gates failed. Observability alone has not solved under-pacing. The retained anchor/reference-demand audit also exposes potentially conflicting targets and a17.78-to-1 pace-loss scaling mismatch in a simplified same-state comparison. This is a hypothesis, not the actual optimizer's exact solution or proof that the public heuristic guarantees physical feasibility.

Pacing-loss weighting is implemented as an explicit bounded training parameter, default1. A paired current-source coefficient1/100 study exactly reproduced the earlier control actor and passed both single-seed development gates. Fresh coefficient100 seeds3001/3002/3003 reached26/28/28 with1/0/0 regressions, so only two pass and replication is still negative. The remaining regression is a position-tolerance failure, not permission to relax that tolerance.

Full-case component diagnostics identified translation residuals, not a reason to freeze them. Optional positive finite pose-budget retention units now preserve gradients on every action head. Fresh precision seeds4001/4002/4003 all reached28/32 with zero success/contact regressions, but efficiency improvements3.76%/7.90%/3.77% leave two below the unchanged5% condition; time gains do not satisfy it either. Replication is still negative and no favorable subset is selected.

Next, separately test efficiency/pacing-loss balance while retaining measured precision, original physics/controller/reward, all four learnable actions and every gate. Do not freeze steering, substitute constant pace, tune one failing task, relax tolerances/efficiency or install a deployment projection. Require fixed-update development, every fresh training seed and a new exposure-bound independent final with frozen candidates and original family mix. Obstacle perception/planning, game learning and hardware calibration remain separate contracts; Nemotron, LocateAnything and AMD cannot substitute for measured movement improvement.

Latest balance studies: coefficient20 fresh seeds reached28/28/27 with all efficiency gates passed but one controller-success tie; coefficient50 fresh seeds all reached28/32 with zero regressions, but one effort reduction was4.9911758%, still below5%. Neither supplies every required gate. Parameter refinement must be predeclared, retain every failure and achieve a real margin before a new independent final; rounding or selecting favorable seeds is not an alternative.

Coefficient40 replication also failed (28/28/27, only one full gate), so continued scalar tuning is not a demonstrated stability fix. Saved training-anchor distribution measurements suggest examining stochastic exploration/credit assignment: about7.27 mm translation standard deviation,9.09 mrad heading standard deviation, and one strongly negative last-batch critic explained variance. These are not final-distribution proof. Next variance/concentration ablations must preserve full stochastic support, every learnable action, exact collected/replayed log probabilities and existing independent-evaluation requirements.

Optional stochastic concentration is now implemented and checked through six focused tests. A current-source fixed-update64 factor-1/factor-8 paired study exactly reproduced the control actor and reached28/32 without regressions for both, with7.7761%/7.8956% effort-proxy reductions. This is one development seed, not a demonstrated stability improvement. Fresh seeds8001/8002/8003 are predeclared for factor8 at the same fixed update/cases/gates. Require every seed before a new exposure-bound final; numerical reconstruction uses the same sampling distribution within float32 arithmetic, while exported-policy resumption remains byte-identical.

Factor8 replication completed: seeds8001/8002/8003 all reached28/32 without success/contact regressions and5.7621%/7.4508%/8.4506% effort-proxy reductions. Every original development gate passes. The new exposure-bound128-case final retained the original8-short/16-field/8-full-contract mix and every fixed candidate/gate, but reached learned79/78/78 versus controller77, neutral80 and initial71, with0/1/1 success regressions and4.6561%/6.7335%/6.2580% effort reductions. Every strict final gate fails; the consumed corpus is retained and excluded from subsequent finals. Phase6 remains open.

An aggregate coverage audit found only4/6-second training deadlines and±40-inch starts, while the public final profiles include10 seconds and±50-inch full-contract starts. Optional `mixed-full-reach-v5` now covers public short/field/full families,6/10-second deadlines and all six physical ranges after four easy starts. Three focused tests pass, including original-engine acceptance and exact resumption. Next is the separately declared current-source V4/V5 pair: fixed seed201/update64, exact V4 actor reproduction, unchanged settings/gates, original32 plus new mixed64 development cases. Require every development suite and fresh training seed before a new final; the coverage hypothesis is not proven and no individual consumed-final task is tuned.

The V4/V5 pair completed with exact V4 actor reproduction. Both reached28/32 and43/64 without success/contact regressions. V4/V5 effort reductions were7.8956%/12.4644% on original32 and6.3058%/9.1852% on mixed64; both original gates pass for both pilots. Fresh V5 seeds9001/9002/9003 are now separately declared and training at fixed update64. Require every seed on both complete suites before another exposure-bound final; the earlier negative final remains retained and excluded.

V5 replication completed: every seed reached28/32 and43/64 without success/contact regressions. Effort reductions were11.4994%/11.5101%/7.6357% on original32 and7.5917%/8.0370%/7.4647% on mixed64; every unchanged development gate passes. However, the new128-case final reached learned67/68/67 versus controller69, neutral70 and initial57, with3/2/2 success regressions despite qualifying effort reductions7.0243%/5.8570%/5.9419%. Every strict final gate fails. Both failed finals remain consumed, retained and excluded; phase6 is open.

Aggregate failure categories span position/heading/settling speed on field/full tasks at both deadlines. Optional periodic measured retention is now implemented with three focused tests: refresh the training-only anchor on the first world of every selected PPO rollout, charge all teacher queries, and commit rows only with complete PPO collection. Default0 preserves the old method. The predeclared interval0/4 pair fixes seed201/update64, V5 and all other settings/gates, with exact disabled-actor reproduction and both original32/mixed64 suites required. This is a new training-state coverage hypothesis, not independent proof; no consumed-final task is rerun or individually tuned.

Periodic retention's paired pilot reproduced the control but reached27/32, tying controller27 rather than improving it; mixed64 retained43 without regressions. The original gate fails, so it is not sent to replication/final. Next, optional `long-history-deadline-v1` supplies16 public frames (565 inputs) with unchanged raw34/action4 and width32. Six focused tests pass, including real Python/JavaScript trajectory parity and exact resume. A current-source4/16-frame seed201/update64 pair is declared with refresh disabled and all other settings/cases/gates fixed; input-layer capacity also grows, so this is a combined history/capacity hypothesis. Require exact control reproduction, every gate, fresh seeds and a new exposure-bound final. Phase6 remains open.

The longer-history pair reproduced the four-frame control but again reached27/32 and43/64 without regressions. The original success gate fails, so no replication/final follows. A saved-control optimizer audit found mean approximate KL up to1.0845 and clip fraction0.5127. Optional phase-specific PPO actor rate is now implemented/tested, preserving imitation0.001 and critic0.0003 while permitting positive PPO rates0.00005–0.001. Next is the predeclared current-source0.001/0.0003 seed201/update64 pair on four-frame V5 with every other setting/case/gate fixed. This is an optimization-stability hypothesis, not proof of causation or independent improvement; require control reproduction, every fresh seed and another exposure-bound final.

The actor-rate pair completed with exact default actor reproduction. Both reached28/32 and43/64 without regressions; reduced-rate effort gains10.7880%/9.0208% pass both development gates but remain below control12.4644%/9.1852%. Maximum approximate KL rose6.0931 while clipping fell0.3119, so stability is unproven. Fresh reduced-rate seeds10001/10002/10003 are predeclared and training at fixed update64 on both complete suites. Require every seed/gate before another independent final; preserve every negative result and keep phase6 open.

Reduced-rate replication completed: every seed reached28/32 and43/64 without success/contact regressions, with effort reductions10.2192%/8.0227%/11.8029% and7.7957%/9.0229%/6.0989%, respectively. Every unchanged development gate passes. However, the new128-case final reached learned73/72/73 versus controller74, neutral77 and initial66, with1/2/1 success regressions despite qualifying effort gains5.7969%/6.7402%/5.2893%. Every strict final gate fails. All three newer negative finals remain consumed, retained and excluded; independent improvement and completion remain unproven.

Analytic KL measurement and an optional collected-state optimizer bound are now implemented and tested. Diagnostic-only reproductions preserve both actors exactly; maximum analytic means2.38905/14.75106 differ from sampled estimates. The seed201/update64 bound0.03 pilot respected the bound and adapted after imitation but reached27/32 and43/64, failing the original success gate. No replication/final/promotion follows. Five optimizer tests and the full117-test Python suite (115 passed, two ROCm skips) pass;51 relevant Node tests pass. Training-distribution stability is not independent movement proof.

Next: inspect aggregate retained development settling margins and whether training distinguishes all four unchanged success criteria. Preserve all four heads, positive learning rates, optimizer/RNG transactionality, raw sensors, original Simulator/iraLIB/reward/tolerances and both complete development suites. Every fresh seed and a new exposure-bound final remain required. No inference projection, favorable subset or consumed-final tuning is allowed. Bounded tactical strengthening follows movement proof; UX/voice/avatar remain planning-only.

Aggregate mixed64 evidence found15/43 learned successes within10% of a tolerance boundary versus12/42 controller successes, without re-simulation or inspecting new finals. Optional V4 now adds one bounded training-only terminal bonus for the worst normalized position/heading/speed/yaw error, preserving V3/full returns and original success tolerances/canonical reward. Five focused real-Simulator/resumption/metadata tests pass. A same-source V3/V4 seed201/update64 pair is predeclared with failed KL/refresh options disabled, all other settings and both complete suites fixed. Require exact control reproduction and every unchanged gate before fresh seeds or another final; this is a reward-sensitivity hypothesis, not proof that margin feedback solves generalization.

Current V4 implementation validation:122 Python tests (120 passed, two ROCm skips),81 focused Node tests passed after approved local temporary-file/loopback access, generated configuration and whitespace checks passed. Paired training is running; phase6 remains open.

The V3/V4 pair completed with exact control/common-warm-start actor reproduction and real subsequent PPO adaptation. Both reached28/32 and43/64 without success/contact regressions. V4 effort gains11.8317%/8.9606% pass both single-seed gates but do not exceed control12.4644%/9.1852%. Fresh V4 seeds11001/11002/11003 are predeclared and training at fixed update64; every seed must pass both complete suites before another exposure-bound128-case final. Preserve all negative evidence and original gates; phase6 remains open.

V4 replication completed at28/28/28 on original32 and43/42/43 on mixed64, with zero success/contact regressions and every efficiency gate passed. Seed11002 ties controller/initial42 instead of improving success, so replication fails; no final is reserved or policy promoted. All63129 accepted rows/99753 actual decisions/625 worlds and every result remain retained. A training-anchor-only audit identified public near-goal normalized XY magnitudes around0.01 versus time/pace0.48–0.90, without final replay or a causal claim. Next predeclare sensor-only precision normalization, retain raw34/action4/four-frame history and every learnable head, test Python/JavaScript parity/source/RNG resume, then require both full suites, every fresh seed and another untouched original128-case final. Phase6 and subsequent bounded tactical strengthening remain open; UX/voice/avatar remain planned-only.

Optional settling-unit normalization is now implemented with unchanged145-input/four-frame/width32 capacity and encoded observations. Only six per-frame units change; public sensor goalXY/velocity/yaw/heading-sine feedback is strengthened numerically, while initial field context and cosine are retained. Increased early-state clipping is a known representation risk, not a proven remedy. Metadata/scale tampering fails closed. Four focused Python and four Node tests pass; full126-test Python discovery passed124 with two ROCm skips, including complete-trajectory parity and exact resumption. Full Node regression is running. The current-source old/new-unit seed201/update64 pair is predeclared with V4 feedback and every other setting/case/gate fixed; exact old-unit actor reproduction, real PPO adaptation, every fresh seed and a new original128-case final remain mandatory. Phase6 is open.

Full sequential Node validation completed:399 tests,397 passed, two GNU Make/Java availability skips, no failures; independently checked inventories include all43 available test files. Combined current implementation receipts are525 tests/521 passed/four availability skips, with generated configuration and whitespace checks passed. Control training is running; phase6 is still an unproven movement-improvement gate.

The paired normalization study completed with exact old-unit actor reproduction and actual subsequent PPO adaptation. Control reached28/32 and43/64; normalized reached27/32 and43/64 with zero success/contact regressions and9.9969%/9.6142% effort reductions. Original32 success ties controller27, so the pilot fails and no replication/final/promotion follows. On identical512-row retained training-teacher tensors, local units clipped5.2829% of input entries versus0 previously, including54.4922% of current goal-heading-sine entries. This confirms a representation risk, not causal final-failure proof.

Next predeclare old145/capacity-matched coarse-augmentation169/local-augmentation169 arms: retain all old field-scale channels and add bounded sensor-only local channels with unchanged raw34/action4/four-frame/width32 contracts. The169-input arms must share capacity/initial parameters/bounded transform and differ only in unit scales; do not confound additional capacity with a pure normalization claim. Test full Python/JavaScript trajectory parity, exact checkpoint/RNG/source resume and every learnable head before training. Require both full suites, every fresh seed and another exposure-bound untouched original128-case final. Independent movement improvement and subsequent bounded tactical strengthening remain unproved; phase6 stays open and UX/voice/avatar stay planned-only.

Both169-input profiles are now implemented: retain base35 and append six bounded tanh channels, with unchanged four-frame/raw34/action4/width32 contracts and1536 additional actor+critic parameters versus145. Coarse/local arms share initial parameters/shapes/scales and differ only in added-channel units. Four focused Python/four Node tests pass, including exact real-Simulator resume, every actor row updated, full heavy-physics trajectory parity and replay. Full130-test Python regression passed128 with two ROCm skips; Node regression is running. The fixed seed201/update64 three-way study is predeclared. Require exact145 actor reproduction and every arm retained; only local-unit multiscale is eligible for advancement if both full gates pass, followed by every fresh seed and another exposure-bound original128-case final. Phase6 remains open; no favorable arm selection or pure normalization claim from extra capacity is allowed.

Full Node regression completed at403 tests/401 passed/two availability skips, without failures. Current combined receipts are533 tests/529 passed/four genuine availability skips; generated configuration and whitespace checks pass. All three declared arms are training sequentially with terminal-state/reference-reproduction guards, not restarting on observation expiry. Movement improvement remains unproved.

The three-way study completed with exact145 reproduction, identical169 initial parameters and actual post-imitation adaptation. Both169 arms reached27/32 and43/64 without success/contact regressions; local effort reductions11.1898%/9.6893% do not replace the original success tie against controller27. No replication/final/promotion follows, and the advancing arm is not switched. Matched512-row teacher tensors retain all coarse channels/context/targets exactly with0% clipping in every arm; local channels are bounded, but60.9863% exceed absolute0.99. Representation correctness has not established improvement.

Next diagnosis is predeclared on retained updates8/32/64 for145/local169, including every case on both development suites and reusing retained64 results unchanged. Distinguish measured imitation-only8 from PPO32/64 without training another candidate or selecting a favorable checkpoint. No diagnostic checkpoint may be promoted or replace the original64/fresh-seed/independent-final gates. This audit is running; phase6 and subsequent bounded tactical strengthening remain unproved, with UX/voice/avatar still planned-only.

Phase diagnosis completed with all12 results retained: original32 success25→27→28 for145 and25→27→27 for local169, controller-success regressions2→0→0 for both; mixed64 stayed43 at every stage, with zero contact regressions. This does not establish uniform PPO degradation or justify selecting imitation-only8. No checkpoint/arm is promoted, and no final/fresh-seed replication follows the failed local64 pilot.

Next predeclare longer fixed96 training in new source-bound runs within existing100-update/50000-decision/900-second CPU bounds, preserving V4/V5/factor8/teacher/anchor and all full suites/gates. Reproduce retained64 prefixes without modifying historical manifests, and choose the advancing architecture before96 results rather than selecting the best checkpoint after inspection. Require every fresh seed and another original128-case untouched exposure-bound final before movement proof and bounded tactical strengthening. This is a data/convergence hypothesis, not an achieved improvement; phase6 remains open and UX/voice/avatar remain planned-only.

The96-update two-arm pilot is now predeclared and training in new directories, with every prior64 result preserved and both byte-exact64 actor/initial prefixes required before96 performance inspection. Original145 is preselected for potential advancement from measured phase progression, not an easier task/gate;169 is retained as the alternate, never chosen afterwards. Both retain all four actions and every original32/mixed64 case/family/physics range. Keep existing50000-decision/900-second bounds and V4/V5/factor8/teacher/anchor/source unchanged. Judge only96, then require every fresh replication seed on both suites and another original128-case exposure-bound one-shot final. Runtime validation remains533 tests/529 passed/four availability skips, not an improvement claim. Phase6/tactical strengthening remain open; UX/voice/avatar remain planned-only.

Both96 pilots completed with exact64 prefixes and preserved historical manifests. Original145 reached28/32 and43/64 without regressions, with11.4848%/8.9832% effort reductions; every pilot gate passes but success has not increased over64. Alternate169 reached28/32 and42/64, so the mixed success tie fails despite qualifying efficiency. Preselected145 seeds14001/14002/14003 are separately declared and training at fixed96 on both complete suites under existing budgets. Require every seed/gate before another original128-case exposure-bound one-shot final; no checkpoint/architecture switch, rounding or policy promotion follows pilot success. Independent improvement and bounded tactical strengthening remain unproved.

Replication is terminal96/96/95;14003 exhausted the original50000-decision cap with112 uncommitted decisions accounted. No cap increase,95 substitution or seed omission is allowed for this study. Resource-diagnosis-only comparison reached28/27/28 on original32 and43/42/42 on mixed64, including one completed-seed14002 success regression. Even completed candidates fail strict performance, so extending resources alone is not a demonstrated remedy; no final/promotion follows.

A source/run/actor-hash-bound training-only diagnosis includes all27 worlds at collections64/80/95 under the exact pre-update actors. Recorded stochastic execution succeeds19/27, deployed deterministic mean16/27, with three one-direction success flips. Python and JavaScript deterministic replay agree on every reason, with maximum canonical-reward difference1.7473e-8. Next predeclare a separately labeled deterministic-actor/twin-Q learning study that directly optimizes deployed mean action values, not a favorable PPO checkpoint or mislabeled replacement. Learned Q functions estimate return only; actual physics remains original Simulator/iraLIB. Require sensor-only inputs, all four trainable reference actions, genuine original-engine gradients/rollouts, exact source/RNG/replay accounting, predeclared adequate resource budgets and every unchanged full-suite/fresh-seed/original128-case independent-final gate. This alignment hypothesis is not proven causation or an achieved improvement; phase6/tactical strengthening remain open and UX/voice/avatar remain planned-only.

Separate TD3 Beta-mean actor/twin-Q learning is now implemented through the unchanged original Simulator/iraLIB. Explicit algorithm tags prevent a PPO claim; Q networks estimate return, not physics. All four references remain learned, with sensor-only145 input features by default, actual replay/target/optimizer states, intrinsic deadline terminals and exact transactional RNG/cost/exposure accounting. Five kernel/four real-Simulator integration/two Node decoder tests pass; full139 Python/405 Node receipts passed137/403 with four genuine availability skips and no failures. Fixed seed201/update64 PPO/TD3 comparison is predeclared and training sequentially, retaining original32/mixed64 gates and exact PPO actor/initial-parameter reproduction. Only TD3 may advance after both full suites, every fresh seed and another original128-case exposure-bound final; no learner/checkpoint switch, consumed-final tuning, physics substitution or policy promotion from implementation tests is allowed. Phase6/tactical strengthening remain unproved; UX/voice/avatar remain planned-only.

## Current movement experiment — 2026-10-10

Current outcome: the standard workspace-only original-engine CLI reservoir full4/prefix2-resume4 probe passed exact actor/critic/target/optimizer/replay/RNG/history/work comparison without rerunning denied tempfile tests. Fourteen current mathematical/compatibility kernels passed. Fixed64 ring control reproduced exactly; reservoir pilot passed original32/mixed64 with11.1401%/7.9431% effort-proxy gains. However fresh17001/17002/17003 reach28/28/27 and42/43/42, with zero regressions and qualifying effort reductions but multiple higher-success ties. Every-seed replication fails, no final is prepared/consumed and no policy is promoted.

The audit now scans the standard TD3 directory as well as recursively nested research outputs. A workspace metadata probe verifies default/nested records and rejects seed/geometry aliases; the added external-temp regression and current full suite remain pending, never relabeled passed. This audit-only change happens after the failed study and does not rewrite frozen historical identities. Future candidates still require genuine reliable deployment-success improvement, all full development/fresh-seed gates and a new untouched original128-case one-shot final before bounded tactical strengthening. Product voice/avatar/UX remain planned-only.

Latest source update: the terminal-sampling pilot fails original32 success and is not advanced. Fixed all-terminal-group regression can fit retained behavior labels but has a large leave-end-group-out gap; added local critic channels do not improve that small cohort. Optional uniform-reservoir replay is now implemented, default recent ring unchanged, with four workspace-only kernel tests and historical storage-diversity rehearsal passed. Real-Simulator resume/work tests, current full regression, exact current-source ring control reproduction, full pilot/fresh-seed/final proof remain pending. Elevated-approval review is unavailable from account usage limits; do not bypass it or claim pending tests passed. The previous551-test receipt belongs to the prior source version. No actor is promoted or final consumed; UX/voice/avatar remain planned-only.

The fixed64 one/eight-step TD3 pair completed: both reach27/32 and43/64, so the original32 success tie fails. One-step actor reproduction and all-row adaptation checks pass; every negative result is retained. Full current regression is547 tests/543 passed/four genuine availability skips, not learned-improvement proof.

Read-only complete-replay gradients motivated a predeclared auxiliary-weight study, not a causal claim. With eight-step credit fixed, scale teacher-anchor10->1 and public-deadline prior40->4; every other original engine/interface/objective/teacher/action head/resource/gate stays unchanged. The control reproduced exactly. The candidate passed both single-seed full development gates:28/32 versus controller27/initial25 and43/64 versus controller/initial42, zero success/contact regressions,9.5686%/6.8539% all-world effort-proxy improvement. Neutral successes28/32 and42/64 remain separate comparators; no policy is promoted from one seed.

Fresh seeds16001/16002/16003 completed64/64/64. Original32 reaches28/28/28 with zero regressions, but seed16002 effort gain4.8547% and time gain0.7006% both miss5%. Mixed64 reaches43/43/42; seed16003 ties controller/initial42. The complete replication therefore fails, despite stronger original32 consistency. No final is prepared/consumed or policy promoted, and no rounding, seed omission or checkpoint switch is allowed. Full-replay diagnosis finds Q/anchor gradient conflict and persistent terminal-value errors, not proven causation; next test value-fitting/credit stability rather than assuming another weight reduction is a remedy.

Optional critic-only terminal-horizon sampling is implemented with inverse probability correction1/(N*q), preserving the expected uniform-row loss/unclipped gradient. The actor/public priors remain independently uniform; default0 preserves existing RNG/math. Configuration/source/replay/optimizer/target/RNG binding and actual original-engine interrupted/resumed learning are tested. Full current regression passes551 tests/547 passed/four genuine availability skips. No new physics/query, actor truth, frozen action axis or changed success criterion is introduced.

A current-source fixed64 seed201 pair preselects mixture0.25 versus uniform, with original eight-step/anchor1/prior4 settings and every full gate/resource bound fixed. Require exact historical uniform actor reproduction, identical initial parameters/units, all-row learning and all original32/mixed64 cases. Critic-target fitting and terminal sampling remain hypotheses, not demonstrated movement improvement. Every future candidate still needs all full development gates, every predeclared fresh seed, then a new untouched exposure-bound original128-case one-shot final with8 short/16 field/8 full families and all original gates. Freeze every learned/initial actor before reveal; any failed gate stops advancement. Only demonstrated independent movement improvement supports subsequent bounded tactical strengthening. No tolerance change, favorable subset, physical/AMD result or publication is claimed. UX/voice/avatar remain planned-only.

## Optional perception track — preparation added 2026-10-02

Latest motion update2026-10-10: the fixed64 TD3 pilot genuinely trained every actor row but reached27/32 and43/64 with zero success/contact regressions, effort gains11.4185%/8.2256%. Original32 ties controller27, so no replication/final/promotion follows. PPO control reproduced exactly. Full replay/anchor gradient norms wereQ0.004614 versus deadline prior0.092292/anchor0.048115, and exploratory-behavior return discrepancies about1.926 RMSE, explicitly not current deterministic-Q ground truth or failure causation.

Optional complete-episode multi-step credit1–32 is implemented, default1 preserving old training. Bind exact horizons, start actions/sensor states/end states and discount exponents; never cross success/fault/intrinsic deadline/reset boundaries. It aggregates actual rewards without extra queries, actor truth, changed original physics or acceptance gates, with explicit later-behavior off-policy approximation. Three independent formula/replay/real-resume tests pass; full regression is running. Next predeclare fixed seed201/update64 one/eight-step comparison, exact one-step actor reproduction, every full suite/gate, every fresh seed and another untouched original128-case exposure-bound final. Preserve all four learned references and every negative outcome. Phase6/tactical strengthening remain unachieved; UX/voice/avatar remain planned-only.

At the user's request, LocateAnything is prepared for future visual target grounding, not enabled as a movement policy or used to bypass the main training phases. See [perception contract](../roboproof/perception/README.md).

- [x] Define versioned image/frame/model-output provenance and parse the model's native normalized boxes/points, with strict malformed/truncated-output rejection and no invented confidence.
- [x] Implement an offline reviewed floor-point + fixed-plane calibration gate producing the shared original-Simulator reach task. Bind image/pose/profile identity, freshness, expiry, held-out measurement declarations, coverage and field bounds; prohibit automatic box-center navigation.
- [x] Expose a read-only preparation status and dashboard section. Keep model runtime, capture, cloud calls and execution endpoints disabled; provide an explicitly synthetic fixture, not fake live inference.
- [ ] Review intended-use/license constraints, custom model code, pinned weights and compatible deployment resources before model setup.
- [ ] Build a bounded inference worker and independently verified image/calibration provenance; demonstrate real grounding on held-out images and explicit latency/ambiguity handling.
- [ ] Add reviewed/revalidated task approval and original-Simulator playback with reachability/obstacle gates. Camera-click navigation and physical deployment remain unavailable.

This optional track does not block coordinate-based motion learning. LocateAnything supplies visual target evidence, Nemotron coordinates supported tasks, the future policy learns movement, and iraLIB executes bounded control in the original Simulator.

## Agent Skills coordination track — 2026-10-06

The existing local Nemotron coordinator now loads five reviewed workflows: inspect robot configuration, prepare a reach-pose experiment, diagnose one explicitly selected saved experiment, review saved PPO/frozen-evaluation evidence and advise on game tactics from reviewed rules/optional explicitly shared state. [Skill contracts and usage](../roboproof/skills/README.md) describe the bounded native tools, on-demand instructions, permission checks and saved hashes. This extends the full project; it does not introduce an independent simulator, robot library or cloud agent.

Skills supply procedural context, not weight training. Movement proposals still require separate approval and execute through original Simulator/iraLIB. Diagnosis and learning review are read-only, keep measured facts separate from model commentary and do not tune, retrain, rerun evaluation or promote a policy. The negative 6/24 improvement result remains unchanged.

Next coordination gate: predeclare multilingual workflow-selection and evidence-grounding cases, measure failures/latency, and retain ambiguity/injection failures. Deeper time-series tools need their own measured tests. This track cannot replace meaningful AMD execution, learned movement generalization, physical calibration, full-game mechanics or camera calibration.

## Tactical hierarchy and movement experiment update — 2026-10-06

- [x] Add a read-only tactical adviser using the existing Override rule/scoring implementation and an explicit compact original-Simulator snapshot. This is not automatic game execution, unrestricted PDF ingestion, complete legal verification or a trained tactical policy.
- [x] Add earlier randomized-physics coverage while retaining easy and heading/direction worlds (`mixed-reach-v2`), with separate PPO actor/critic, exploration, KL/clipping and curriculum diagnostics. Original physics, reward, controller gains and frozen evaluation gates remain unchanged.
- [ ] Before extending training, test action feasibility and reward ablations on development-only worlds, then reserve a new untouched final corpus. Do not tune against inspected frozen cases and relabel them fresh evidence.
- [ ] Define typed game subtasks, legal/feasible prerequisites, reliable mechanism skills, strategy evaluation, approved dispatch and replanning before a tactical AI can lead specialist modules autonomously.
- [ ] Complete camera/calibration, genuine AMD acceleration and physical transfer independently. PID/LQR and scripted manipulation are not trained sub-AIs.

Detailed architecture and gates: [tactical AI plan](TACTICAL_AI_PLAN.md).

## Product UX, voice control and embodied assistant track — planned, not implemented

The project will eventually have a single, understandable AI workspace rather than exposing separate technical labs as the primary user experience. This section is a product plan only; it does not authorize implementation or claim that voice control exists.

### User experience goals

- Make the home screen feel like a focused robotics AI application, with one conversation area, one live robot/simulator status area and clear task history.
- Replace the current expert-oriented navigation as the default entry point with a guided main screen. Keep Motion, Perception, Tactics, Experiments and Evidence as advanced views.
- Show the selected execution target explicitly: original Simulator, connected VEX robot or planning-only mode. Never imply that a command ran on hardware when it only ran in simulation.
- Show the assistant's interpretation, planned steps, safety checks, execution state and final result in plain language.
- Provide visible cancel, pause, emergency-stop and fallback-to-controller controls before any command can move a robot.

### Voice and conversational control

- Add microphone input with recording, permission, transcription, language selection and clear failure states.
- Add text input as a fully equivalent fallback; every voice command must be reviewable and editable before execution when it can move the robot.
- Connect the assistant/avatar (the planned animatronic character) to the conversation state, task progress and simulator feedback. The avatar is presentation, not an authority over physical state.
- Support direct bounded commands such as `move forward 24 inches` and `turn right 90 degrees` by converting them into typed motion intents with units, frame, speed and timeout.
- Support compound tasks such as `go to this point`, `pick up a Pin` and `place the Cap there` through a structured task plan, not a free-form motor response.
- Confirm ambiguous commands, impossible targets, missing calibration, unavailable mechanisms and hardware-risk actions instead of guessing.

### Command pipeline

```text
Voice/text input
    -> speech-to-text and language normalization
    -> intent parser with typed units and coordinate frame
    -> permission, feasibility and safety validation
    -> tactical/task planner
    -> perception and target grounding when needed
    -> motion planner / learned policy
    -> bounded PID/LQR/iraLIB execution
    -> original Simulator or explicitly selected hardware target
    -> telemetry, score, explanation and saved evidence
```

- The language model may interpret, explain and plan, but it must not emit unvalidated motor voltages or bypass the existing controller.
- Every command receives a correlation ID and records the transcript, parsed intent, approvals, target, plan, actions, telemetry, score and errors according to the local evidence policy.
- Simulation execution must use the same original Simulator and control interfaces as training and evaluation. Browser UI must not create a second physics loop.
- The assistant must expose whether a result is planned, simulated, replayed, learned, or physically executed.

### Planned UX/UI phases and gates

1. Audit the current frontend and group controls into Home, Simulator, Motion, Perception, Tactics, Experiments and Evidence.
2. Define the typed command/task schema, units, coordinate frames, target modes, permissions and error vocabulary before adding a voice button.
3. Build a text-only conversational prototype against a mock executor and test interpretation, ambiguity and safety states.
4. Connect text commands to the original Simulator for bounded commands: forward distance, rotation, target pose and reviewed pick/place tasks.
5. Add voice transcription with visible transcript editing and confidence/ambiguity handling.
6. Add the animatronic/avatar presentation, task timeline, live telemetry and result cards without allowing it to conceal errors or execution mode.
7. Add tactical planning for multi-step game tasks, with human approval before execution and replanning when the simulator reports failure.
8. Add hardware mode only after calibration, safety interlocks, controller fallback and supervised tests pass.
9. Test keyboard, text and voice parity, responsive layout, accessibility, localization, reconnects, cancellation, stale commands and simulator/hardware labeling.

**UX gate:** a new user can understand the execution target, ask for a bounded task, review what the AI understood, stop it, and see an honest result without opening internal training labs.

**Voice-control gate:** the same typed intent produces the same validated action plan whether entered by keyboard or voice; ambiguous or unsafe speech never becomes an automatic movement command.

**Product gate:** a short end-to-end demo shows `speak -> understand -> plan -> approve -> move -> report` in the original Simulator, with the baseline and learned-policy result clearly distinguished. This is separate from, and cannot replace, the independent movement-improvement gate.
