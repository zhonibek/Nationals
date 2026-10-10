# RoboProof Agent Skills

Five reviewed workflow packages extend the existing local Nemotron coordinator. They supply procedural context and restricted native tools, not new model weights or a separate simulator. The original Simulator, iraLIB execution path and separate PPO learner remain authoritative.

| Skill | Native tools | Result |
| --- | --- | --- |
| `inspect-robot` | `get_robot_profile`, `finish_analysis` | Nominal configuration, source hash and controller-path explanation; not hardware identification |
| `prepare-motion-experiment` | `get_motion_contract`, `prepare_reach_pose` | Bounded single-target proposal; the existing separate approval runs the original Simulator |
| `diagnose-motion` | `get_saved_motion_evidence`, `finish_analysis` | Analysis of one user-selected saved run; measured facts remain separate from model commentary |
| `verify-motion-improvement` | `get_learning_summary`, `finish_analysis` | Read-only checkpoint/frozen-evaluation summary, preserving all seed comparisons and failed world IDs |
| `plan-game-tactics` | `get_game_rules`, `get_game_snapshot`, `finish_analysis` | Reviewed Override rules and optional explicitly supplied Simulator state; conditional advice, no dispatch or game-policy claim |

## Try it

Use the existing [local runtime setup](../NEMOTRON.md), then open [Chat with Nemotron](http://127.0.0.1:8766/#nemotron) and expand **Robot tools**. No new model download is required for an already configured installation. The default **Send** action is ordinary conversation without tools; it cannot read project evidence or prepare an executable task.

1. The five skill cards load through `GET /api/nemotron/skills`. Opening the page does not invoke Nemotron, train, simulate or access hardware.
2. **Explain robot & controllers** asks the model to activate the inspection skill and read `config/robot.json`. This is a nominal snapshot; its PID configuration must not be confused with the simulated LTV-LQR control core.
3. **Review learning progress** invokes read-only analysis of current checkpoint hashes and saved evaluation. Changed weights do not imply accepted improvement. Missing or stale evaluation is reported as unavailable.
4. **Explain latest Motion lab result** explicitly selects the latest saved Motion lab session. To analyze a particular approved Nemotron task, use **Explain this result** underneath its measured result. If no completed result exists, run an experiment first.
5. Enter a coordinate request and explicitly click **Prepare simulation from my message**. This produces a proposal, never automatic execution. Review the human-readable coordinates (exact JSON is expandable), then separately click **Approve & run in original Simulator**. New approved runs record transitions for later diagnosis; old reports without transitions remain usable but expose that absence.
6. **Explain game rules & tactics** discusses reviewed rules without claiming current state. For snapshot-backed advice, open the original Simulator through RoboProof, start Override and click **Спросить тактический AI**. This explicitly shares a compact, structurally validated browser snapshot, not independently verified telemetry. It cannot become an executable task; [tactical architecture and remaining gates](../../docs/TACTICAL_AI_PLAN.md).

Ordinary chat accepts natural-language questions, including Russian, and saves its own conversation separately. Use the named robot-tool buttons when you need actual configuration, learning evidence or a selected saved result rather than general model commentary. Tool replies are separate from conversational history. Model understanding and explanations still need review. Sparse telemetry supports observations and hypotheses, not proven root causes or full oscillation analysis.

## Loader and security

Packages use the public [Agent Skills format](https://agentskills.io/specification): directory name, `SKILL.md`, descriptive frontmatter and workflow instructions. This implementation intentionally supports only five reviewed packages and single-line `name`, `description`, optional `compatibility`, and `allowed-tools` metadata. It is not a general YAML parser, arbitrary third-party skill installer or script executor.

Only catalog metadata enters the initial prompt. The model calls `load_skill` to load the selected instructions; at most two packages may activate within a six-turn session. Each skill is limited to 12 KiB, rejects symlinks and metadata/tool mismatches, and is checked against its catalog SHA-256 before loading. Instructions and `allowed-tools` cannot expand the hardcoded runtime permissions. No shell, arbitrary file paths, uploaded scripts, remote provider, source edits, motor voltages, firmware upload, training or automatic policy promotion are available.

Native function calling is required on every turn of the separate skill workflow, not ordinary chat. `ask_clarification` safely ends an ambiguous or unsupported request without a task. `prepare_reach_pose` is not offered until the contract has been read; `finish_analysis` is not offered until real evidence has been read. Unsupported plain-text claims of preparation are rejected, not turned into tasks. These phase gates are executable runtime checks, not just advice in a prompt. Provider failures are not retried automatically.

After activation, only selected instructions remain in system context; unused catalog descriptions are removed and full skill text is not duplicated in tool messages. Provenance still retains the complete reviewed instructions and hash in the saved tool log. This reduces prompt overhead, not the multi-GB model's weight-memory requirement. On a busy 6 GB Windows host, live requests can still page heavily or time out: release resources before starting the runtime or use a sufficiently provisioned local compatible endpoint. Do not treat fixture tests as a completed live model-quality benchmark.

Diagnosis uses an explicit API selection `{source: 'motion' | 'nemotron', id: '<saved UUID>'}`. The server reads that saved session and constructs a bounded snapshot before inference. The model cannot select another report or provide a fabricated report in place of it. At most five sensor/evaluator samples enter the snapshot; full transitions are not sent to the model. The report hash, recorded replay flag, final-state settling checks and measured metrics are retained. A previous exact-replay flag is not a fresh verification, and final-state checks do not reconstruct 15-tick dwell.

New sessions save activated skill names/hashes, permitted tools, usage, model commentary and tool evidence to the existing ignored `roboproof/runs/nemotron/` directory. Compact JSON files are limited to 16 MiB, including approved-run telemetry. They survive server/browser restarts and can be downloaded; prompts and evidence may be private, so review before sharing. Model API keys remain server-side. Existing loopback, same-origin, response-size, timeout, duplicate-call, task-validation and separate-approval safeguards remain enforced.

## Validation and next steps

`tests/agent-skills.test.js`, `tests/agent-skills-ui.test.js` and `tests/nemotron.test.js` cover progressive loading, hashes, permissions, malformed requests, selected evidence, negative/stale evaluation, persistence, explicit approval, original-engine execution and exact replay. Mocked tool responses are fixtures, not proof of live inference. The UI-event tests use a fake DOM; they do not validate actual browser appearance or GPU performance.

Next: evaluate skill selection and grounded explanations across a predeclared set of multilingual requests, ambiguity and injected instructions. Add deeper time-series investigation only with measured tests and bounded tools. Broader tasks, learned movement improvement, full-game physics, camera grounding, physical calibration and meaningful AMD execution remain independent product gates.
