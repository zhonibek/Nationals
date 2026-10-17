# Experimental GVF in the original control stack

Updated 2026-10-16. This is an optional analytic movement controller, not a trained GVF policy or an independently qualified improvement. The old controller remains the default.

## What is connected

- `include/subsystems/control/GuidingVectorField.hpp` uses the existing Pedro `BezierCurve::project` and geometry. Each reference depends on measured pose/velocity, not elapsed trajectory time. Fixed feedback gives the same reference after a pause; the actual action deadline still applies.
- The field supplies a curve projection, a tangent velocity and bounded cross-track attraction to the existing Riccati-based C++ `Cascade`. That cascade and its persistent wheel PI/feedforward remain the only motor-control path. The legacy Pedro follower is not enabled as a second writer.
- Both production WASM and the hardware `followGuidingField` entry point use this same header. The hardware entry point keeps the existing output lease, sensor/odometry-epoch checks, cancellation, timeout, competition-mode handling and release-to-zero. It is not automatically enabled on a physical robot.
- Original Simulator actions use estimated odometry and wheel feedback, not evaluator truth. Motion Lab offers **GVF + existing LQR · experimental, not learned**. Its normal saved-evidence and exact-replay paths work with this mode.
- Success still requires the actual goal pose and low measured velocity, not merely being stationary somewhere on the curve. Evaluation rewards, physics, deadlines and qualification thresholds are unchanged.

## Fixed v1 bounds and supported paths

The retained v1 comparison below remains historical. Current working-tree v2
adds an explicit GVF-only 1.5 rad/s bound on the **combined** Cascade angular
request, not just field feedforward, and requires sensor heading error below
0.005 rad before release. The ordinary controller retains its previous unbounded
request path and unchanged numerical golden; wheel/voltage safety limits and
all evaluation gates remain unchanged. Both shared hardware and Simulator call
the explicit bounded path. This limits requested velocity, not guaranteed true
yaw or physical stability. Fourteen lightweight checks and hardware-double safety
checks pass locally; the local HTTP API test cannot connect under the current
sandbox network restriction. The subsequent complete remote motion/API comparison
is recorded below. No independent qualification or learned improvement is claimed.

The field reference has a 0.35 m/s translation cap, attraction gain 1.5/s, 0.6 m/s² curvature speed budget and braking budget, and a 1.5 rad/s heading-feedforward cap. Curvature from Pedro is converted from inverse inches to inverse meters. Acceleration is not added to velocity as if the units were interchangeable. The existing wheel loop still applies its own acceleration, wheel-speed and 12V limits; these are not claims of measured physical acceleration or universal stability.

The first implementation supports lines, stationary heading-only commands and cubic paths whose control-polygon edges progress monotonically along the endpoint chord. It rejects retracing/closed nonstationary paths and unsupported cusp geometry instead of accepting ambiguous projection equilibria. Repeated start/end control points are supported with a bounded tangent fallback. This is a deliberately explicit geometric domain, not an arbitrary spline planner, obstacle avoidance system or route optimizer. Reach-task commands construct collinear cubic control points from the measured start to the requested goal.

## Compatibility and evidence

The original scripted reference and existing four-action policy adapter stay unchanged. Existing actor checkpoints are **not** GVF actors: GVF options reject learned identity, including through the public wrapper. The old pace/action contract is not silently reduced to three actions. New source/WASM identities mean historical reports/checkpoints are historical; they are not automatically current or exact-replayable against changed sources.

Twelve focused local checks cover projection, bounded outputs, heading-only commands, actual endpoint completion, invalid input, retracing/cusp rejection, repeated endpoints, old actor rejection, sensor-only feedback, persistent controller state, short exact replay and the normal Motion Lab API. The tests simulate only short 0.1-second episodes; they do not establish full-path success. A 25-row point/Bezier fixed-feedback golden from the preserved pre-GVF WASM remains byte-identical numerically. Production C++ boundary tests exercise timeout, settlement, cancellation, busy ownership, sensor failure and changed odometry epoch using hardware doubles, not a physical robot.

The pre-change source/WASM copies are retained in ignored `.cache/pre-gvf-20261016`. Rebuild with `tools/build-wasm.ps1` before running new control sources; never edit manifest hashes to bypass stale-build checks.

`tools/evaluate-gvf.js` declares every historical frozen-v2 case before execution, evaluates the unchanged controller and fixed GVF on **all** cases, exact-replays each, and preserves failures, all-world metrics, regressions and source hashes. Run that campaign only on an approved remote Linux host, with `ROBOPROOF_APPROVED_REMOTE=1` and an unused output directory. This comparison does not train or promote a model and is not a newly untouched final. Long regressions and all learning remain remote-only.

## What is not done

1. Establish actual controller-level results across the full fixed comparison and disturbances before claiming GVF helps. Do not tune on a consumed frozen test and then call it independent.
2. Specify a separate versioned residual-learning observation/action contract for GVF, with no clock pace, measured limits, reward audit and dataset provenance. Keep the current actor/teacher corpus in its original contract.
3. Generate new teacher/native movement data on the original Simulator remotely, then train and retain all independent seeds/checkpoints/exposures. Match or explicitly revalidate runtime/source changes.
4. Recover or conservatively bound the earlier lost Colab exposure chain before a new independent final. The unresolved loss registry remains active; preparing a GVF mode does not clear it.
5. Pass unchanged independent comparisons against both the existing controller and own untrained policy before promotion. Physical timing, calibration, slip recovery, obstacle planning and game scoring remain separate work.

No claim is made that GVF guarantees faster PPO convergence, perfect recovery from collisions, a 2–4× sample-efficiency gain, game-score optimization or learned improvement. Those require measurements.

## Actual first remote comparison

The foreground Colab job completed all three stages on 2026-10-16 at 17:23:14 UTC: 12/12 focused checks, the existing 17 closed-loop controller regressions, and every historical frozen-v2 world for both controllers with exact replay. Actual host: Linux, Python 3.13.15, Node 20.19.0, two CPUs, allocated Tesla T4 15,360 MiB. This run used the remote CPU for original Simulator/WASM; no GPU inference or model training occurred.

| All 24 historical cases, failures included | Existing controller | GVF v1 |
| --- | ---: | ---: |
| Successes | 6 | 5 |
| Failures | 18 | 19 |
| Mean elapsed seconds | 7.64625 | 7.77875 |
| Mean effort proxy, V·A·s | 39.159938685964455 | 48.716618729175586 |
| Mean final position error, m | 0.31336828925568144 | 0.1692160536527431 |
| Mean contact seconds | 0 | 0 |

GVF regresses two controller-success cases: `frozen-v2-5-nominal` and `frozen-v2-5-power-contact`, both heading-only tasks. It newly succeeds on one other case. The two regressed tasks end stationary with true heading error approximately 0.0564 and 0.0559 rad, above the unchanged 0.035 threshold. Smaller mean position error is **not** a pass or a reason to hide these failures. The current fixed GVF is not better overall and is not promoted. These historical cases are now inspected for this method; they are not an untouched independent test for later revisions.

Complete result archive: 67,536 bytes, SHA-256 `5c47d9862e3a8a1cc4f360f09d72d9d38322dcb399ca0068f24b8f82381c8f0c`; all 38 result members and CRC verify on the computer. Source transport: `9317d5ff5cf8d0ea0eda54a8b4f8722a2fa00dcf45568775a675138b4c084949`; actual controller WASM: `0885404c31c17c3b14e579bfa8d443a4c8efc7e791d73d52f90dfdb8fbfcb1fa`. Retained original reports live under ignored `roboproof/runs/motion/research/gvf-fixed-v1-20261016`, with all declared worlds available to the exposure collector. The result archive is also in the approved private RoboProof Drive folder; connector readback confirms its size, folder and owner-only permissions, but does not expose a remote checksum. The source archive stays on the computer/authorized private Colab, not in that results-only Drive backup.

The dashboard option is added and its run/save/replay API passes; visual browser verification on localhost was blocked by the saved browser permission. No workaround was used and no rendered-UI pass is claimed.

## Actual v2 turn-repair comparison — 2026-10-17

One bounded foreground Colab job completes at 2026-10-16 22:13:22 UTC
(2026-10-17 local time). Actual host remains Linux/Node20.19.0, two remote CPUs,
allocated Tesla T4 unused by this analytic Simulator study. All15 focused tests,
including the real Motion Lab HTTP save/replay test, and the original17-scenario
controller regression pass. A separate three-test local metadata suite verifies
the fixed declaration, source/WASM binding and refusal of unauthorized campaigns.

The evaluator predeclares all16 inspected development turns plus all24 historical
worlds, executes both controllers on both preserved/current source variants,
and exactly replays every160 report:320 actual episode executions including
replays. The ordinary controller's full numeric reports remain identical across
both source variants on every40 world. No parameter search, training, new final,
gate change or automatic promotion occurs.

| All failures included | Existing controller | GVF v1 | GVF v2 |
| --- | ---: | ---: | ---: |
| Development-turn successes /16 | 14 | 2 | 6 |
| Historical successes /24 | 6 | 5 | 5 |
| Total successes /40 | 20 | 7 | 11 |
| Mean effort proxy across40, V·A·s | 30.965684163009445 | 42.75636339052114 | 34.75814409993427 |
| Mean contact seconds | 0 | 0 | 0 |

The combined angular request limit is actually observed: new turn wheel-target
yaw never exceeds1.5rad/s and no development-turn output clips12V. There are no
new v1-success or contact regressions. However v2 still regresses ten successes
of the existing controller, including both historical180-degree failures, so it
is **not** better than the existing controller or eligible for promotion.
The improvement relative to faulty v1 is not neural-policy improvement.

The retained failed development turns finish internally settled with estimated
heading error about0.0044–0.0046rad while true error is0.0384–0.0687rad.
Tighter completion alone does not remove wheel-odometry/truth disagreement.
Truth is used for evaluation only; it is not added to controller observations.
Do not erase the failures, retune on consumed historical worlds or relax the
0.035rad external gate. Sensor/model limitations and a separately versioned
residual-learning contract remain open before new training.

Complete results:127,679 bytes, SHA256
`181771c20ed1968f60796195cb45141064e4f4ad5325459487f0d19506c75344`;
all58 payload hashes/CRC verify locally. Transport SHA256
`24061557f0610ba12aa20b3cc2cdfc92947e1358e9c01529bb2818274fc6263c`,
actual v2 WASM `457f5fe1088cc325ac12b6909f88647c6401265a92c8e243ad4678d9a934b7a7`,
summary SHA256 `7e07921ec1438cfd8b09c9f8d27602e57a597e242d06d0d0ce810964cbc8026b`.
Every original case and both declarations are retained under ignored
`roboproof/runs/motion/research/gvf-v2-turn-repair-20261017`, so inspected
development seeds/geometries remain available to the normal exposure scan.
The archive also has verified size/parent/owner-only metadata in the approved
private RoboProof Drive folder; an unavailable remote checksum is not claimed.
No training remains running. The earlier missing independent-training event
remains unresolved; this comparison does not admit its recovery envelope.
