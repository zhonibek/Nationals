# GVF residual-learning integration

Status: episode/bridge/replay wiring and separate neural-model definitions are
implemented; actual PyTorch execution and learning remain unverified. No trained
GVF actor or qualified policy exists. The
existing scripted controller remains the default. Existing four-action PPO/TD3
contracts, checkpoints and teachers must not be relabelled as GVF policies.

## Distinct control contract

The new contract is explicitly named `gvf-residual`, version2. Its three bounded
actions request small body-right, body-forward and heading reference corrections.
There is no pace action or advancing trajectory clock. GVF itself remains the
existing analytic Pedro/C++ reference generator, followed by the original
LTV-LQR and wheel controller. This does not create a replacement physics model.

Zero residual is `[0,0,0]`, not the legacy four-action default. The native GVF
pose/velocity reference and sensor-derived progress remain separately visible
from the filtered correction. Repeated observation reads must not advance either
the residual filter or a virtual clock. Only measured odometry/wheel feedback,
public task/deadline, native geometry and adapter state may reach the actor.
Truth pose, physics configuration, contacts and reward/evaluation components
remain outside actor inputs.

The separate module `simulator/gvf-policy.js` now implements this contract and a
sensor-only34-feature adapter. It initializes the actual native `gvf` interface
once, filters the three corrections, preserves raw native headings at zero
residual, and exposes detached observations. `refreshSensors()` is an explicit
post-physics sampling boundary: it updates measured feedback and the native field
sample without advancing the filter, changing the already issued command or
resetting the cascade. `observe()` and `vector()` remain pure.

Fourteen focused adapter tests use a mocked native export buffer through the
real JavaScript control wrapper. They cover the distinct schema, SI/inch boundary,
zero-action identity, fresh sensors, saturation/anti-windup, reset and malformed
inputs. They do not execute the real WASM or Simulator physics. Subsequent
focused checks below exercise the actual native boundary separately; there is
still no trained GVF actor or movement-improvement claim.

An equal observation-vector length never establishes compatibility: names,
versions, feature semantics, action shapes, normalizations, source/controller
identity and checkpoint algorithms must all verify explicitly. Old time/pace
demonstrations cannot simply lose one action column and become new teachers.

## Native execution boundary

The original Simulator now recognizes a distinct persistent
`gvfResidualReference` action. It resets the original cascade once on activation,
validates its complete finite six-value reference, and explicitly invokes the
native `gvf` stepping path. That path retains the combined LQR/feedforward yaw
request limit, unlike the ordinary legacy learned-reference path.

New corrected positions remain inside ±1.524m per field axis, translational
feedforward inside0.35m/s in magnitude and yaw feedforward inside1.5rad/s. These
are reference-request limits, not guarantees about physical chassis velocity,
acceleration, stability or obstacle clearance. Existing timeout and controller
fault handling remain. No new motor or physical-robot access is introduced.

Persistence is deliberate: an uncorrected GVF-complete flag must not release a
learned residual action before the unchanged episode termination rules. Therefore
persistent zero-residual GVF is a separate comparator from the existing analytic
GVF runner, which releases its action after its own completion. Any resulting
improvement from persistence alone is not learning.

Four local engine-boundary tests use small controller mocks only: explicit native
mode/persistence, unchanged legacy mode, malformed/sparse/excessive references,
and existing timeout/sensor-fault neutralization. They execute no physics and do
not establish real-WASM, training, replay or performance success.

## Implemented episode and bridge path

`MotionEpisode` now dispatches only an explicitly tagged new contract. Its
five-tick macro-step, ten-tick watchdog, stop/fault handling, original rewards,
deadlines and15-tick settlement rules remain explicit. A post-physics refresh
prevents stale actor inputs. Terminal command age is measured in physics ticks,
not filter updates; a stale command can terminate without advancing its filter.
Native sample/sensor failures produce an unsuccessful terminal and a labelled
last-valid observation fallback, not invented current measurements.

Node runners expose `runGvfPolicy()` and a separate
`runGvfZeroResidual()` comparator. Node replay and browser-playback logic validate
the distinct contract. The new module is included in headless source identity,
browser script order and the server's static-asset allowlist. Old source-bound
reports remain immutable; a different source identity is not silently accepted.

The bridge selector is process-fixed: default legacy `policy`, or explicitly
`--contract=gvf-residual`. Python uses
`MotionEnvironment(control_mode="gvf-residual")`. The handshake checks the
schema/version, all declared sensor fields, action shape/names, limits and fixed
cadence. No time-trajectory reference duration is supplied for GVF. Invalid
actions stop and require reset; repeated terminal stops consume no extra physics
tick. A mismatched reset neutralizes the active command. A failed stop closes the
Python bridge rather than continuing with unconfirmed control. New-mode
`motionTrainingImplemented` remains false: wiring an environment is not a learner.

### Actual focused native checks — 2026-10-17

These are permitted lightweight focused tests, not laptop training or simulation
campaigns. The Node check uses one fixed straight task with a0.03-second deadline
and one three-value action: exactly three original Simulator physics ticks, then
three ticks of exact Node action-only replay and three ticks through the browser
playback implementation in Node. A separate actual Python/Node bridge check
executes three ticks. The current source passes both checks; the earlier initial
source check and its exact641,509-byte source packet are retained separately.
The full total across both source revisions is21 physics ticks on the same
predeclared world, with source-bound reports and conservative exposure saved
before querying. This is not teacher collection, an optimizer step, a performance
campaign, a successful reach task or independent qualification.

Actual WASM remains
`457f5fe1088cc325ac12b6909f88647c6401265a92c8e243ad4678d9a934b7a7`.
Each tiny episode ends at its declared time limit. The evidence proves the
three-action plumbing and replay work on that short case, not that GVF is better,
a neural model was executed, or rendered browser UI was verified. Large native
campaigns and all training remain remote-only; Colab reopening is still unverified.

The current frozen source packet is684,828 bytes, SHA256
`1f50d2fcccc81bdb07c174c3b5931918c8b32b1e2ce026dde2c6a461311a4542`.
The18-payload results-only archive is25,225 bytes, SHA256
`32bb593cc7c53548fee9226bfa0e3a723cc39a9a4a599c9222cd9a4fa620ed9b`.
All payload hashes and CRC verify locally. Private Drive readback also verifies
the expected size, parent and owner-only/unshared permissions; it exposes no
remote checksum. These backups do not establish a trained actor or qualification.

## Separate neural model preparation — 2026-10-17

The new `gvf_contract.py`, `gvf_model.py` and `motion-gvf-learner.js` define a
distinct three-output tanh actor and two independent Q critics. The artifact
algorithm is `td3-tanh-gvf-residual-v1`, not the previous four-action Beta actor.
Hidden widths32/64/128/256 and depths2/10 are explicit architecture choices, not
evidence that more layers improve movement. The actor's final layer initializes
to zero, giving a neutral residual before learning; its own initial weights must
still be frozen and preserved before an actual fit.

The feature transform `gvf-body-reference36-v1` starts only from the explicitly
tagged34-channel sensor observation. It rotates measured translation velocity,
reference displacement/velocity, applied offsets and offset velocity into the
measured body frame. Absolute measured position remains for field context.
Measured heading and relative reference heading use sine/cosine pairs, giving36
features rather than a hidden trajectory clock. Target-error channels are already
body-relative. Fixed declared float32 scales and clipping to[-10,10] use no fitted
statistics, true pose, physics parameters, contacts or evaluator components.
Remaining public deadline is context, not a pace action or success classifier.

The separate JavaScript reader validates exact contract/algorithm/architecture,
normalization, three-output dimensions, finite bounded weights, original engine
identity and optional exact artifact SHA. Reading is byte-bounded and verifies
file stability. A caller can explicitly run an artifact through the persistent
GVF runner/replay; the module does not enable a dashboard default or promotion.
An artifact's `trained` metadata is not proof of gradient training or improvement.

Nine tiny JavaScript synthetic/reader tests and six pure-stdlib Python tests
pass, including exact scales and actual tagged feature calculations across both
languages. They do not execute PyTorch, optimization or Simulator physics. Four
additional tiny verification-driver guards pass. Five PyTorch model/export/
nonzero-actor parity checks are deliberately skipped on this laptop and require
the approved remote Linux host. Their existence is not a remote pass.

The separate `gvf_td3.py` now implements the gradient core: twin-min targets,
bounded target-policy smoothing, a delayed actor update every second critic step
and Polyak target updates. It has no pace head, legacy demonstration conversion
or changed reward. The caller must mark every intrinsic deadline/fault/truncation
as done; gamma discounts one five-physics-tick decision. Only explicit float32
CPU or indexed CUDA devices are supported, with no device fallback. Optimization
rejects a non-approved/non-Linux host before tensor work.

Core state preserves its four networks, both Adam states, update counters and
Torch CPU/selected-device RNG under exact contract/architecture/config/device/
Torch-version checks. This is not a complete fit checkpoint: native environment,
replay, sampler, Python/NumPy RNG, source/runtime identity and exposure custody
still need the fit driver. A failed partial optimizer update cannot continue or
be published as a resumable checkpoint until a verified state is restored.
Eleven remote-only synthetic gradient/continuation/failure tests are present but
unexecuted here. An independent static review also identifies mixed Boolean/list
promotion in tensor conversion; pre-conversion rejection and remote regressions
are added without loading Torch locally.

The final focused local batch passes44 JavaScript and19 Python mock/stdlib tests;
all16 actual Torch model/gradient checks are skipped before importing Torch.
The batch explicitly confirms Torch is absent from the process. No new native
physics query, optimizer step, trained artifact, fit or performance verdict is
produced by these checks. Existing episode/bridge mocks remain distinct from
the earlier21-tick native evidence, which retains its original source/scope.

`tools/verify-gvf-learning.py` prepares a bounded foreground remote verification
job, requiring actual approved free Colab/T4 inspection and a frozen source
manifest. It records the actual runtime, source hashes, all outcomes and complete
logs, including failures. Synthetic gradient checks are labelled separately from
movement training, and the allocated T4 is explicitly unused by CPU unit checks.
No paid fallback, native campaign, actual fit, final preparation or promotion is
performed by this driver. Results still require verified local/private custody.

The user's approved ordinary-Chrome route is attempted after the in-app browser
fails. Browser discovery has no connected Chrome surface; native discovery finds
one Chrome New Tab window, but the first window-state action stops because the
browser URL cannot be determined confidently enough to enforce safety policy.
No bypass, Colab upload, runtime allocation or training follows. The user must
open the Colab page manually before another legitimate inspection can proceed.

This does not solve the measured wheel-odometry/true-heading disagreement or add
an independent localization sensor. A neural residual cannot guarantee recovery
from disturbances its inputs cannot distinguish. Actual new measured data,
remote gradient/continuation/export checks and unchanged movement gates remain
necessary before claiming useful learned adaptation.

## Required before training

- Complete and remotely verify the separate three-action optimization/fit/
  checkpoint path around the new actor/normalization/export definitions, with a
  frozen own-untrained actor. No weight slicing or fabricated warm-start parity.
- Verify full episodes, Python/JavaScript actor parity, checkpoint resume and
  fault/replay behavior on the approved remote original Simulator/WASM. The tiny
  native checks above do not replace that scope.
- Collect new measured native teacher/online data remotely, retaining all queried,
  failed and interrupted exposure. Resolve compact-ledger capacity losslessly
  before larger fits; do not discard prior history or relax its budgets.

## Qualification remains unchanged

Every declared evaluation world must compare the trained actor against the
original controller, analytic GVF, persistent zero-residual GVF and its own frozen
untrained actor. Retain the existing strict gate: higher success than the original
controller and own untrained actor, no controller-success/contact regressions,
at least5% all-world mean time or effort gain, and three independent training
seeds. Extra comparators improve attribution; they do not replace this gate.

The latest40-world analytic study has11 GVF-v2 successes versus20 for the existing
controller and ten controller-success regressions. It is not a promotion or
learned improvement. Wheel-odometry/truth heading disagreement remains measured;
this software contract supplies no independent localization sensor. Sample
efficiency, superiority and successful future learning must be measured, not
inferred from fewer actions or combining controller names.
