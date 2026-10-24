# Rules-grounded tactical AI inside RobotAI

## Product goal

Given a reviewed game rulebook, robot capabilities and observed match state, propose a strategy, assign bounded subtasks to specialist modules, measure their outcome and replan. Rules explain what earns points and what is prohibited; they do not teach grasping, calibrated motion, opponent prediction or winning play by themselves. Existing Simulator/iraLIB remain the execution and evaluation foundation.

### Current training scope — 2026-10-16

The intended game benchmark is to earn more actual simulator points, not merely reach more poses. This is **not yet the implemented learning task**. Current PPO/TD3 experiments train bounded reach-pose movement: position, heading, settling, time, contact and an electrical-effort proxy. Game points, possession, pickup and placement are absent from that learning contract. The game has its own scoring engine and scripted manipulation; neither becomes a learned scoring policy by displaying a motion replay. Nemotron tactics remain read-only advice.

The prior critic-burn-in method fails all-seed replication: only one of its three independent runs passes both unchanged suites. A separate episode-balanced replay pilot passes both development suites, but its later independent evidence chain is incomplete after Colab runtime loss. The prior pilot is preserved locally and in the approved private Drive folder; unavailable independent exposures now block new final preparation instead of disappearing from the ledger. No new final is prepared and no policy is independently qualified or promoted. See [the current movement validation record](MOTION_LEARNING_VALIDATION_20261016.md). More neurons, GPUs or repetitions cannot by themselves connect the missing game task.

Before score-oriented training, the original Simulator must provide a reproducible full-game reset/step/report contract, bounded real mechanism actions and observed object/possession state. Verify acquisition, transport, placement, failed attempts, timing, legality and score conservation against the existing engine. Freeze a game-specific reward based on actual engine score changes and separately report terminal score/differential, penalties and task outcomes; do not count predicted points, repeated pickup credit or an isolated movement reward as scored points. Start with a measured scripted scoring baseline, then compare the learned game policy on unseen layouts/physics/opponents under a separate predeclared game gate. Keep the movement gate unchanged and report movement and scoring qualification separately. These are dependencies, not claims that a game-training adapter or trained scoring policy exists.

```text
Versioned, reviewed rules + game observation + robot capabilities
  -> Tactical coordinator: choose conditional priorities / team roles
  -> Future typed subtask queue + legality / feasibility / approval gates
       -> Motion: original iraLIB control + experimental residual PPO
       -> Manipulation: existing lift / grasp / place, later learned skills
       -> Perception: future calibrated LocateAnything observations
  -> Original Simulator / scoring engine -> measured result -> replan
```

These are specialist roles, not a claim that every component is already a trained sub-AI. PID/LQR are controllers. Nemotron is pretrained; procedural skills do not train its weights. Manipulation is currently scripted. Vision inference is disabled. Independent movement qualification remains unpassed; a development pilot is not a qualified policy.

## Implemented first slice

- A fifth reviewed `plan-game-tactics` skill reads the pinned Override 2026–2027 v2.0 simulator rule subset, source hashes, scoring constants, goal geometry and capability limitations. `get_game_rules` and `get_game_snapshot` are read-only tools; `finish_analysis` requires both rules and snapshot availability to have been read.
- In the original Simulator served through RoboProof, start Override and click **Ask tactical AI**. Only this explicit action captures and sends a compact game observation to the local model. It includes phase/time, selected robot, all four reported poses/possessions/DQ flags, reported score, goal-top/stack summaries, Toggles, resource counts and at most six nearby Pins/four Cups. It does not upload full telemetry, code, raw images or the complete rulebook.
- The snapshot has an exact schema, fixed game/version/IDs, bounded numbers/size, capture freshness and phase consistency checks. Only the standard head-to-head world variant is accepted; other event variants require their own reviewed rules. It is user-supplied browser state, not independently authenticated or replayed ground truth; nearby objects are not complete field visibility. It can be outdated by the time inference completes. Source hashes prove which server rule implementation was supplied, not correctness of every imported state value.
- Snapshot-backed requests may activate only the tactical skill. They cannot become reach tasks or approve execution. Advice and rule/snapshot provenance save in existing Nemotron sessions; the original Simulator displays readable text and Stop cancels inference. The coordinator's Robot tools also has a general rule/tactics explanation action when no snapshot exists.
- This is an adviser, **not** an autonomous match player, exhaustive referee, verified score forecast, trained game policy or execution dispatcher. It never changes game state or drives robots. Ordinary chat remains separate and cannot read the current board. No cloud/GPU provisioning occurs.

Rule source: [VEX public Override v2.0 complement](https://www.vexrobotics.com/override-manual), checked 2026-10-06. The [official PDF](https://link.vex.com/docs/26-27/v5rc/game-manual) has priority; it could not be retrieved during this update. The small paraphrased rule subset follows the existing engine, not an exhaustive manual transcription. Human adjudication and missing rule coverage remain necessary. Review use/redistribution terms before ingesting or distributing full manuals.

## Gates before autonomous tactical leadership

1. **Rule ingestion and review.** Pin the authoritative rule revision and source hash, retrieve only relevant sections, map scoring/legal constraints to typed checks, and require review for updates or a different game. A random uploaded PDF must not automatically unlock commands or alter scoring.
2. **Feasible subtask contracts.** Version tasks such as acquire, carry, place, set Toggle, park and hold. Define prerequisites, possession/mechanism limits, phase/goal restrictions, deadlines, result types and cancellation. Disable every unsupported skill. Include geometric routing/obstacle checks instead of assuming a direct reach target is a route planner.
3. **Original-engine strategy evaluation.** Compare bounded strategy candidates or a search policy in the existing four-robot game, with separate opponent policies and untouched evaluation matches. Measure actual score differential, invalid attempts, task completion, regressions and timing. Predicted points never replace the scoring engine; incomplete contact physics and referee-dependent cases remain disclosed.
4. **Approved dispatch and replanning.** A deterministic orchestrator owns the queue, resource locks, deadlines, stale-state checks and emergency stop. The LLM chooses only allowlisted objectives; it cannot generate shell/firmware scripts or bypass controllers. Start with separately approved simulator-only programs, not physical autonomy.
5. **Learned tactical policy.** Only after reliable game-task observations/rewards/actions exist, evaluate demonstrations/search distillation or hierarchical RL. Do not relabel the current reach-only PPO as a scoring or opponent-strategy model. Profile CPU rollout throughput before moving larger batched training to genuine ROCm hardware.
6. **Camera and physical transfer.** Add calibrated perception, identified mechanism/drive dynamics and physical safety tests independently. Simulation score is not evidence of real-world performance.

## Parallel movement-learning improvement

The CPU PPO trainer now uses `mixed-reach-v2`: four easy starting episodes, then repeated direction/heading, randomized-physics, easy, randomized-physics episodes. This guarantees two harder worlds in the first eight completed episodes while retaining easy coverage, instead of waiting until episode 24 for physical variation. Bounds, controller gains, reward, success criteria and frozen test worlds are unchanged.

New checkpoints record curriculum identity and stage counts, separate actor/critic losses, differential entropy, approximate KL, clip fraction, critic explained variance and sampled action mean/std/boundary fraction. These diagnose training; they do not prove improvement. Continuous differential entropy may be negative, and explained variance is null for constant returns. No timing is inserted into deterministic replay histories. Old checkpoints cannot resume across trainer-source/curriculum changes; their measured evaluation remains historical evidence, not a new result.

Next movement steps are a development-only action-feasibility probe, reward ablations for heading/settling, bounded longer multi-seed runs and an untouched final corpus. They are not silently completed by the curriculum/diagnostic changes. No new winning policy is claimed.
