# RoboProof known issues

Updated 2026-10-02. Scope distinguishes canonical reach-task preparation, legacy failure research, and separate physical/game/cloud gates. A bounded CPU reach-task experiment does not require finishing camera or whole-game features.

| Priority | Issue | Evidence / next action |
| --- | --- | --- |
| P0 | Motion-policy training not implemented | Original browser/headless engine, persistent bounded actions, Python bridge, complete transitions, stored replay and readiness smoke are implemented. Next implement genuine CPU gradients, policy checkpoints/resume and independent before/after evaluation. |
| P1 | Legacy experiments remain a distinct plant | Scalar tests, failure-predictor ML and old tensor parity still use `legacy-reduced-v1`; they are explicitly labeled and never pooled into canonical motion transitions. |
| P1 | Harder canonical evaluation exposes failures | Frozen v2 controller baseline succeeds in 6/24 worlds, nominal scripted adapter in 7/24. These are retained stress results, not learned improvement or physical reliability. Do not tune the final corpus, loosen success tolerances or hide long/deadline-pressure cases. |
| P0 | Deep ML has local readiness, not validated AMD/search benefit | Actual CPU training and guarded acquisition exist. Canonical datasets/region holdouts, meaningful model evaluation, repeated matched-budget search comparisons and real AMD gradient execution remain required. |
| P0 | AMD provider API not supplied | Job packaging and fake lifecycle are tested, remote transport disabled with zero spend/no retries. Await documented endpoint/access/capabilities and explicit budget before implementing any live provider. |
| P0 | Genuine AMD ROCm execution blocked on this host | CIM identifies Ryzen 3 5300U / AMD Radeon Graphics device 0x164C, driver `31.0.21912.3005`; this APU is not in the [ROCm 7.14.1 support matrix](https://rocm.docs.amd.com/en/docs-7.14.1/compatibility/compatibility-matrix.html). Run strict `--device rocm` on a supported host. AMD speedup: NOT MEASURED. |
| P0 | Full success story has no accepted controller fix yet | The 256-world experiment improves pass count under a candidate but introduces regressions; it is correctly rejected. Broader candidate design requires a supported root-cause hypothesis and the same gates, not threshold changes. |
| P1 | LLM coordination is not movement learning | Local Nemotron task inference is connected and has retained live evidence; default failure diagnosis and `local_agent.js` remain deterministic. External/cloud investigation is separate and unverified. Historic unsupported PyTorch/HIGH-causal labels are warned about in the UI; new diagnosis reports failed sweeps and uncalibrated hypotheses honestly. |
| P1 | Plant fidelity uncalibrated | Effective wheel mass, contact stiffness, drag, motor response and parameter ranges are assumptions. REQUIRES PHYSICAL TEST; collect acceleration, wheel/chassis velocity and turning data before predicting actual reliability. |
| P1 | Java reference absent from this worktree | 91 pre-existing Pedro deletions belong to the user. The unchanged read-only sibling was successfully compared: 2,077 cubic cases, with 177 documented curvature cutoff differences. Reproduction elsewhere still requires the original reference checkout and a compiler-capable JDK; do not silently restore/edit the deleted tree. |
| P1 | Production runtime not fully simulated | C++ control/reference source is shared, but PROS scheduler, position-based odometry implementation, safety leases/watchdog and sensor epoch faults are not. See simulator audit. |
| P1 | Only a point-to-point task adapter | No whole-autonomous routine, collision, obstacle, manipulator or field-game verification. Full Foresight/follower equivalence is not implied by geometry agreement. |
| P2 | PID components unavailable | The existing WASM ABI exports total command and wheel targets, not independent P/I/D/FF terms. Tool reports this explicitly. |
| P2 | Sensor timing quantized | Delay and update periods use ceil(dt), with initial samples and sample-and-hold dropout; sub-tick sensor timing is not modeled. |
| P2 | Report definition gaps | Settling/completion and saturation exist; explicit peak overshoot and calibrated confidence are not implemented. First-divergence detection currently covers path/localization only. |
| P2 | Per-run command/sensor history retained | Scalar long batches may allocate substantial memory; GPU telemetry should be disabled for benchmarks and batches bounded to hardware capacity. |
| P2 | Git process blocked in sandbox-generated metadata | CPU benchmark records `software_version: UNVERIFIED` when child-process execution is denied. Full source and controller hashes remain recorded; run outside that sandbox for Git revision metadata as well. |
| P3 | Universal framework ingestion and polished 3D | Deferred by design; the reference adapter must remain the focus. |

No autonomous physical test is authorized or executed by this prototype. Nothing in the reported robustness score certifies universal robot safety.

Added CI covers Node original-engine/model-tool contracts and Python tensor/auxiliary-ML/motion-bridge tests on Windows/Linux, with explicit genuine-ROCm/Java skips. The workflow has not yet run on GitHub. Existing uncommitted/untracked source remains a separate publication/release task; this implementation does not commit, push, restore Pedro deletions or upload firmware.
