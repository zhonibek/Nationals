# Twenty-agent engineering review

Twenty explicitly requested engineering subagents received distinct jobs. The runtime allowed six at once; they completed in waves and were closed after returning. They are development agents, not twenty robot-control models. The parent owns integration and serialized validation on this 6 GB machine.

## Completed assignments

| Job | Scope | Result |
| --- | --- | --- |
| 1 | Observation safety | Exact 34-slot/sensor-only tests; compensating group-width bug identified and fixed |
| 2 | Reward accounting | Original-engine component, bonus, progress and physical-success tests |
| 3 | PPO audit | Narrow exploration, finite-deadline bootstrap and minibatch weighting findings |
| 4 | Control/odometry audit | Controller math versus firmware safety and physical truth; slip observability limits |
| 5 | Diagnostic UI | Missing/negative/null/sampled-action labels and promotion tests |
| 6 | Holdout audit helper | Exclusive commitment/candidate freeze/one-shot consumption, explicitly not tamper-proof secrecy |
| 7 | Leakage audit | Exact/family/near-duplicate exclusion and lifetime split-role risks |
| 8 | GPU preflight | Isolated strict HIP/CPU capability reporting; no workload/GPU performance claim |
| 9 | ROCm port audit | Collection/device/RNG/source portability plan, not a completed GPU port |
| 10 | Tactical context | Bounded immutable compactors; full provenance remains separate |
| 11 | Tactical legality | Missing observations and advisory versus executable legality distinction |
| 12 | Resource diagnostic | Read-only artifact/Node/RAM metadata; no hard memory or integrity guarantee |
| 13 | Portable setup | Windows clone/start/training/artifact-transfer guide |
| 14 | Firmware audit | Cancellation/ownership/watchdog/stop-confirmation and packaging risks; no hardware validation |
| 15 | Snapshot edge cases | Freshness, phase/variant, possession, visibility and inventory tests |
| 16 | Future game objective | Signed final score differential, legal gates and opponent evaluation design |
| 17 | Evidence boundaries | Nested source/secret/telemetry leakage and stale-comparison tests |
| 18 | CI portability | Linux path, PowerShell execution, generated-file line-ending and source-hash risks |
| 19 | Learning presenter | Pure readable formatter with explicit missing/negative/scoped evidence |
| 20 | Acceptance audit | Current passes/gaps and strict original-engine, independent, tactical, vision and hardware gates |

## Integrated work and limits

The parent tightened observation group widths, separated measured evaluator information from policy inputs, fixed source/run-bound comparison handling, constrained model evidence, strengthened tactical snapshot validation, added strict holdout lifecycle and integrated readable diagnostics. PPO experiments now declare initialization, finite-deadline training objectives and separate actor/critic bodies. Training-reward experiments do not alter original physics, motor gains, action limits or measured pass thresholds.

The first longer development trial (seed 114, 32 updates, 9,800 steps) did not improve success over its untrained actor and regressed two controller-success development worlds. This is negative development evidence, not a fresh final result. Further candidates are selected on development only. Pre-reveal commitments invalidated by source changes are abandoned without reading their cases; final evaluation must freeze all three seeds and retain every failure.

The later seed-131 development result (13/16 learned, 12/16 controller, 13/16 neutral, 10/16 initial) motivated a common 64-update production experiment, not best-seed selection. Production seeds 201/202/203 completed 58,856 total policy steps and 686 episodes. Fresh committed final `b819eb9d-b7d3-4365-bfca-2b46dad58854` then failed every strict seed gate: controller 87/128, neutral 88/128, initial 80/128, learned 75/83/81 with 12/5/8 controller-success regressions. Measured effort-proxy reductions (14.42%/14.60%/17.22%) are not a substitute for improved success. Baseline remains default; no accepted movement-learning product is claimed. All cases and failures remain retained, and this final corpus cannot be reused as fresh evidence.

The latest integrated validation passed 400 of 402 combined Node/original-Simulator tests, with two configured-runtime toolchain skips (GNU Make and the surviving Java reference), and 89 of 91 Python tests with two unavailable-ROCm skips. There were no failures. Precision retention now passes the success/regression part for all three fresh development seeds, but two still fail the unchanged efficiency condition, so overall replication remains negative; detailed identities and all historical failures are retained in [progress](progress.md). Passing implementation tests does not establish independent policy improvement, live tactical quality, calibrated game physics, AMD performance or safe hardware deployment.

Outstanding firmware audit findings must be reviewed/tested before physical deployment: cancellation launch races, inactive legacy writers outside leases, watchdog dependence on sensor/mutex progress, failed-stop confirmation and clean library packaging. They are static findings, not demonstrated hardware faults. No firmware was uploaded, no physical safety certificate exists, and adding a test or compiling firmware is not a safety proof.

Tactics remains read-only and partial-rule based. Vision, calibrated mechanisms, autonomous game dispatch, actual ROCm workloads and physical transfer remain independent gates. No benchmark or model-quality success is inferred merely from these reviews or helper implementations. No changes are published without an explicit request.
