# Remote training and evidence custody

Training, large teacher collection and mass simulator evaluation run only on
approved remote resources. The laptop is for editing, dashboard use, bounded
artifact transfer/integrity checks and focused lightweight tests. No local model
or training fallback is permitted.

## Before a fit

1. Inspect the actual allocation, allowed price/quota, runtime and execution
   device. An old output, GPU selector or provider status alone is insufficient.
   CPU training on a T4-equipped host must still be labelled CPU training.
2. Preserve the exact original source/runtime/data identities and predeclare all
   seeds, fixed budgets, development/final separation and original pass gates.
   Previously missing runs are not unused seeds or new untouched worlds.
3. Verify an approved durable destination. The existing Drive connector can write
   to the user-approved owner-only RoboProof folder without mounting Drive or
   copying credentials into Colab. Notebook and connected-Drive accounts differ;
   do not expand sharing or grant the runtime access to an entire Drive just to
   bridge them. Checkpoint/log/result uploads are authorized, not arbitrary
   personal-file or source-code uploads.

## One bounded foreground fit at a time

Run the existing synchronous trainer in one foreground notebook cell. Do not use
an unattended background shell across turns on an ephemeral Colab runtime. An
observation timeout is not a terminal process: inspect the same exact live handle
before any restart. Do not use browser keep-alive tricks, paid upgrades or quota
workarounds.

GVF training decision/time limits stop new episodes at episode boundaries; an
active original-simulator episode is allowed to finish, with at most 199 decisions
of declared overshoot. This preserves complete, auditable exposures without
silently discarding the fixed decision or wall-clock budget.

Before the next fit or a turn boundary:

- Verify actual terminal state; retain incomplete and failed runs as well.
- Preserve every actor snapshot, checkpoint, complete actual exposure, manifest,
  configuration, runtime/source provenance, stdout/stderr and per-case outcome.
- Read files consistently, bind their SHA256 and verify archive membership/CRCs.
- Transfer the original artifact bytes to the computer, then the approved private
  Drive folder. Respect the connector100MiB file limit with standalone disjoint
  result volumes and a complete catalog; never silently drop older snapshots.
- Read back file names, sizes, parent folder and private permissions. If the
  connector omits remote checksums, record that limitation rather than claiming
  remote SHA verification. Do not start the next seed until custody is verified.

Missing temporary runtime files cannot be recreated by a green status, another
model, a new latest pointer, a refreshed session or repeating the same seed.

## If evidence becomes unavailable

Preserve the last actual observation and current read-only recovery evidence.
Record an unresolved event in `roboproof/motion-exposure-losses.json`, which is
versioned so it survives moving a checkout without ignored runs. Keep the original
study marker, seeds and known run IDs. Do not infer unobserved successes, model
hashes, later process status or whether a reserved third seed started.

Independent final preparation must fail closed until original exposure is
recovered or an independently verified source-bound conservative superset covers
every possibly queried world. Such a superset is exclusion metadata, not measured
queries or a replacement physics model. A self-declared resolved flag does not
establish recovery. Any recovery workflow must retain its evidence and all old
events and must not weaken or redraw consumed qualification cases.

The2026-10-16 lost independent campaign has no qualification verdict. Its prior
proof/control/development pilot remains preserved and retains only its original
scope. Movement qualification and bounded tactical execution remain separate
dependencies, not consequences of successful backup operations.

## Verified conservative exclusion admission — 2026-10-17

The conservative43,008-world plus128-offline envelope and its independently run
source-bound audit are preserved. A new bounded reader can admit only the exact
known event, pinned audit/auditor/input archive, manifest and every original
shard, while keeping the event unresolved and all three training seeds reserved.
It also requires every previously retained historical source/hash and geometry;
restoring only the latest recovery files must not erase older exclusions.

The approved foreground Colab validation completes with exit0 at
2026-10-16 22:59:29 UTC. All29 metadata tests and13 full-data negative cases pass,
including a changed shard/audit, extra member, forgotten historical source,
partial checkout, self-dismissed event, changed-seed/deadline geometry alias,
original offline world and each reserved seed. Full admission retains167,921
metadata records,869 source files,62,488 seed exclusions and60,927 geometry hashes.
The3,125,470-byte lossless snapshot round-trips exactly, and exact restoration
after negative tests produces the same ledger hash. No new physics, model load,
optimization or final draw is involved.

After verifying the complete results locally and in the approved private Drive,
register the exact proven descriptor in the current versioned event. Its bytes
match the validation copy, SHA
`8670000b5f0d74a014c20075d96f63fb124f64865e7add27d86ea89f2ff414a0`.
The event stays **unresolved**, the missing models/outcomes remain unknown and
all three seeds stay reserved. Registration enables the conservative exclusion
path only when every original input and historical-source floor verifies; it
does not certify learned improvement, repair a missing model or prepare a final.

Complete validation archive:2,321,131 bytes, SHA
`fd6ec5f5eb1c06e79b9d0c6f9be8bd17021dfcf756f5a4372be3d383761d58dd`.
All14 payload hashes/CRC verify locally. Private Drive readback confirms expected
size, parent and owner-only/unshared permissions; no remote checksum is exposed.
The isolated validation proof remains unchanged and accurately says it did not
modify production; production registration is a later separately recorded step.

An immutable diagnostic history snapshot preserves818 source references,
124,782 observed metadata records,19,477 seeds and17,920 geometry hashes. It
explicitly reports830 legacy seed-only records and two unavailable-event markers,
and is not a valid independent-final ledger. Its exact SHA is
`40427436d6912e09ae8c81d25a87e2074f2eaa8daed46ba5b3df17c39edda603`.
The admitted ledger preserves those830 legacy seed-only records and adds three
explicit reserved-seed records:833 in total, with
`legacyGeometryFullyKnown:false`. Do not relabel incomplete geometry as fully
known or claim arbitrary historical/OOD separation.

The results-only historical backup retains817 original metadata sources plus
that inspection:25,554,848 bytes, SHA
`22ce1fd7b88367fc82885e648165dfe9ad0a58b4dbc291ff3b94c4f447e3e346`.
No source-code file, model weights, credential or unconsumed final payload is
included. The original source input ZIP and auditor code remain separate owned
project inputs; the results-only Drive authorization does not authorize uploading
those code files. Keep all original pilot/model results and failed runs as well.

Full admission must retain the existing128MiB/4MiB/200,000-record budgets. The
current full-data inputs approach the total byte ceiling; before additional
large fits, plan and independently validate a lossless compact exclusion artifact
or another bounded, source-bound representation. Do not simply raise limits,
discard history, move away required run manifests, or clear the unresolved event.

## Compact exclusion conversion preparation — 2026-10-17

The proposed capsule preserves the complete ordered43,136 seed/geometry pairs,
including duplicates, while retaining original task/configuration bytes in the
exact pinned7,583,049-byte archive. It is a separate exclusion projection schema,
not a complete historical ledger, model recovery or a new final. Production still
uses the previously verified full-shard descriptor; compact admission is not
registered or enabled.

A bounded independent review finds and repairs two issues before execution:
the mapper now pins immutable original normalization/geometry function digests,
and the Python archive reader reopens the parent by pathname to reject directory
replacement. Six tiny JavaScript tests pass. Six tiny Python tests pass locally;
the additional descriptor-relative pathname test is skipped on Windows and must
pass on the approved Linux server. No full-data conversion has run yet.

Python/JavaScript trigonometry is not presumed universally identical at half-bin
boundaries. The remote procedure records a tiny adversarial boundary comparison,
then independently recomputes and strictly compares every actual archived pair.
Any real-record mismatch must fail conversion; do not widen geometry tolerances,
drop records or self-certify the Node projection. Original JSON/file/aggregate
budgets and historical-source requirements remain unchanged.

Reviewed conversion-only source packet:13,364 bytes, SHA256
`bb7b2697000c723adb3ee8882dd2144e977b2a49769c509cf8d878cd5330332e`.
The earlier11,962-byte packet is superseded before execution; its file upload was
not verified. Neither packet contains credentials or personal files. These source
packets are not covered by the results-only Drive backup authorization.

The notebook reports an inactivity/max-session disconnect during an upload UI
timeout; its page subsequently crashes. No conversion cell was inserted or run,
so this is not a lost training job or a new exposure event. Earlier terminal full
admission/GVF evidence is already preserved locally and privately. Automatic
reopening creates an about:blank tab, but its navigation times out on inspection;
do not interpret that as a restored kernel, actual allocation or running process.
Inspect/recover the same attempt before creating duplicates or executing work.
Do not bypass browser security, service quotas or use local training as fallback.

## Fresh-kernel recovery preparation — 2026-10-17

The existing Colab notebook crashes, and a separate attempt to open the empty
Colab home also times out. Its exact new tab remains blank and cannot be read;
this is not proof that all Colab servers are down, a live training process, or a
verified GPU allocation. Lightning's actual organization page still serves its
official credential-security incident notice and says user access is revoked.
Do not reuse saved Lightning credentials, upload project inputs there, bypass
restrictions or allocate paid resources. Ordinary Chrome has been proposed as a
separate user-approved UI route; it is not yet used.

A new local clean notebook and9,543-byte bootstrap packet prepare fresh Colab
recovery without relying on the vanished old kernel. Bootstrap SHA256 is
`4e3a423c33683bf866a14134c12cd606c4b70f30c089c661c39e26a11ea04459`.
Restore only the exact pinned34,267,098-byte original admission source packet,
2,321,131-byte preserved admission results and13,364-byte reviewed capsule source.
Copy the original proof bytes, not a newly generated self-declaration; restore
the previously verified registry only in the separately owned remote workspace.
Every nested manifest member/hash/CRC and canonical contained path verifies.
Conflicting files, symlinks, duplicate/extra members and existing job roots fail
closed. Seven tiny local packaging tests pass; no full restoration or conversion
has executed on the laptop or a new server.

Install the same pinned official uv0.9.2/CPython3.12.3 remotely, preserve logs and
check actual Node20.19.0/TeslaT4 allocation. The operator must verify free resource
eligibility in Colab: the script explicitly cannot certify provider billing.
The existing independent conversion/audit remains a separate foreground cell.
Source/controller/mapper and all original128MiB/4MiB/200,000-record gates remain
unchanged; an actual mismatch stops advancement. Neither restoration nor capsule
creation enables production compact admission or a training fit.

The clean notebook has no saved outputs and no embedded base64 result archives.
Use the normal Colab file download for the result ZIP and verify its original
bytes locally, then preserve results only to the approved private Drive folder.
Do not let another huge output-filled notebook become the only recovery route.

A separate opt-in single-fit draft preserves the original four-action argument
factory instead of trying to train the unimplemented three-action GVF learner.
It binds an explicitly supplied separate frozen historical root, actual runtime,
complete freshly recomputed ledger and independently audited capsule before one
fixed64 fit. It refuses the reserved seeds and does not automatically chain fits,
reveal a final or promote a model. Tiny mocked tests pass; no trainer or model is
loaded locally. This remains preparation, not a ready launch: actual remote/source
admission, compact-capacity integration and a reviewed terminal/custody interlock
retirement procedure are still missing. Do not clear its guard or the old loss
merely to run the next seed.

## Initial AMD GVF source package — 2026-10-17 (superseded)

`tools/package-amd-gvf-training.py` creates a deterministic, allowlisted source
archive for the separate GVF residual TD3 pilot. It includes the original
Simulator/iraLIB runtime and compiled controller, exact controller source
identity, Python training modules, frozen evaluation fixtures and any consumed
holdout exclusion records. It does not include run outputs, private keys,
provider credentials, or arbitrary workspace files. The archive is not a model
result and does not mean training has started.

This initial training-only archive is obsolete; do not upload it. The prepared
local archive was 2,021,118 source bytes across71 files; archive
SHA256 `760e5e7662d54dcbed35c1651893ced001bf5fb3f91eed111059b95f844f4602`,
source-manifest SHA256
`fe7ef11a106fe082858c463b95cb3be3a83df03c8decb39a8c6bed8437e17db5`. Every
manifest source hash verifies against the current checkout, and the ZIP contains
only those71 files plus the manifest. The exposure ledger now ingests every GVF
episode-start event and fails closed on a missing/mismatched JSONL record, so
future independent-final preparation can exclude completed and interrupted GVF
worlds. Its23 focused exposure tests and6 trainer-guard tests pass. The source
manifest reader requires an exact versioned schema, exact rows and lowercase
SHA256s before any simulator import or GPU use. Its import-safety check runs in
a fresh subprocess so unrelated test-module imports cannot affect the result.
The complete GVF-focused discovery run passes61 tests with16 remote-only checks
skipped as intended; a separate three-physics-tick contract smoke is explicitly
not model training or learned-improvement evidence.

No SSH-authenticated server session or optimizer steps are verified. Before
uploading or fitting, inspect the live AMD allocation and remaining approved
credit/time cap, authenticate through the local key prompt without sharing its
passphrase, verify MI300X/ROCm/PyTorch and Node.js on that server, and use this
exact source bundle. If SSH cannot be unlocked, stop the allocation in the AMD
dashboard to avoid idle credit use. A pilot checkpoint, even if later produced,
still requires independent original-simulator evaluation and cannot pass the
movement qualification gate by itself.

## AMD GVF evaluation package — 2026-10-17

The remote evaluator at `roboproof/gvf-motion-experiment.js` is paired with the
training package. The primary candidate is now a 512-wide, 10-hidden-layer
GVF residual TD3 actor, trained independently with seeds50512,50513 and50514.
The prior 64×2/50101–50103 cohort remains readable as historical evidence; it
is not the primary candidate. The evaluator checks exact source manifests,
original Simulator identity, MI300X optimizer use, architecture and seed cohort,
then compares the unchanged controller, analytic GVF with zero residual, each
initial actor and every trained actor. Frozen development and the independent
128-case final keep the existing strict acceptance rules unchanged. No best
seed selection is allowed, and a fresh final holdout can be reserved only if
all three primary seeds pass development. All development/final simulation
entry points reject a Windows/local host unless the approved Linux remote
environment is explicitly set.

The evaluator's development identity and final exposure comparison were fixed
to match the original holdout-audit contract. Tests cover those comparisons
using mock reports only. The source archive now carries the evaluator runtime,
the full recognized historical exposure records and the exact reviewed
conservative recovery envelope; this is required so the server can reproduce
the same exclusion ledger instead of accidentally forgetting prior local runs.
Development and one-shot final evaluations now fsync a bounded append-only
progress journal after each exact-replay episode and completed case. This
retains partial evaluation evidence after interruption; it does not permit a
second reveal or resume that would reuse a consumed final.
The package builder's stdout is authoritative for the exact file count, source
bytes, archive SHA256 and source-manifest SHA256. Rebuild after any source or
documentation change. On the server, calculate the manifest digest from the
uploaded bundle's `source-manifest.json`; do not reuse a hash copied from an
older package or this document.

A read-only collection of the current project ledger verifies 882 source files,
167,934 observed world records, 62,489 unique seeds and 60,927 unique body-relative
geometries, plus 833 seed-only historical records. One earlier unavailable run
remains unresolved; its independently audited conservative exclusion is
preserved and explicitly does not claim model or result recovery. The snapshot
is for this project state and is not a claim about exposures outside the
retained evidence.

Validation:103 focused JavaScript GVF/evaluator/holdout tests pass; the GVF Python guard
suite passes61 tests with16 remote-only skips. The Python test suite also ran its
explicitly labeled three-tick native contract smoke; it is not training or a
qualification campaign. No training or independent evaluation was started, and
no model improvement is verified. The browser confirms GPU droplet `610013899`
is Active at `$1.99/hour`, using the PyTorch 2.14.0 / ROCm 10.1 image. The owner
provided Web Console screenshots showing a root shell at 2026-10-17 17:15 UTC,
low host load (0.04) and memory use (1%), followed by an idle root prompt. The
screenshots do not show a trainer process, GPU utilization, checkpoint or
training result; they cannot establish whether a background process exists.
The image notes that PyTorch runs inside the `rocm` container and host ROCm is
unknown. This session cannot interact with that console directly.
Billing Overview reports `$1.53` month-to-date usage and `$98.47` remaining
account credits, but notes daily updates and does not isolate this droplet's
current charge. No trainer process, MI300X identity, or GPU-enabled PyTorch
runtime is verified; obtain the read-only `amd-smi`/`rocminfo` and in-container
PyTorch checks before upload or training. SSH port 22 is reachable from this
workstation, but its local OpenSSH agent is stopped/disabled and no
`known_hosts` file exists. The existing project key is passphrase-protected;
enter its passphrase only in a local terminal, never chat. Before connecting,
compare the host-key fingerprint reported inside the AMD Web Console
(`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`) with the candidate key
fetched over SSH. Do not bypass host-key verification. No training run or
learned improvement is verified.

After authenticated access to the already-approved Linux MI300X is restored,
extract this archive on that server and run the primary 512×10 cohort as three
separate foreground fits with the same manifest hash. Each fit has a one-hour
wall-clock ceiling, 100,000-decision and 100,000-update caps, and its own
output directory. Check and preserve each completed run before starting the
next; a GPU does not make serial Simulator rollouts or qualification finish in
five minutes automatically. Stop if the approved promotional balance cannot
cover the next run; do not use the payment method or auto-reload.

```sh
export ROBOPROOF_APPROVED_REMOTE=1
export SOURCE_MANIFEST_SHA256="$(sha256sum source-manifest.json | cut -d' ' -f1)"
python -m roboproof.motion_learning.gvf_train --approved-amd-mi300x --approved-max-credits 90.51 --source-manifest source-manifest.json --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" --output roboproof/runs/motion/gvf-td3/gvf-td3-seed50512-w512-d10 --seed 50512 --max-decisions 100000 --max-updates 100000 --max-seconds 3600 --network-width 512 --hidden-layers 10
python -m roboproof.motion_learning.gvf_train --approved-amd-mi300x --approved-max-credits 90.51 --source-manifest source-manifest.json --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" --output roboproof/runs/motion/gvf-td3/gvf-td3-seed50513-w512-d10 --seed 50513 --max-decisions 100000 --max-updates 100000 --max-seconds 3600 --network-width 512 --hidden-layers 10
python -m roboproof.motion_learning.gvf_train --approved-amd-mi300x --approved-max-credits 90.51 --source-manifest source-manifest.json --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" --output roboproof/runs/motion/gvf-td3/gvf-td3-seed50514-w512-d10 --seed 50514 --max-decisions 100000 --max-updates 100000 --max-seconds 3600 --network-width 512 --hidden-layers 10
```

The three 512×10 seeds are the main learned candidate, not a short pilot.
Training completion means the declared fit budgets ended and checkpoints were
saved; it does not mean the movement system is “100% ready.” Promotion still
requires all three seeds to pass frozen development and the same independent
128-case final gate against the unchanged controller, with no success/contact
regressions and the predeclared effort improvement. If training, development,
or the independent gate fails, report that result; do not weaken thresholds or
call GPU utilization, allocated VRAM, or completed optimizer steps a pass.
The 512×10 actor's larger artifact remains inside the explicit 64 MiB reader
budget; its capacity is not evidence of better movement or a reason to fill
the GPU's VRAM.

Run development evaluation first. If any model fails, do not create a final
holdout. If all pass, the evaluator's `prepare` and `final` operations take the
three run folders, the saved development JSON and one new holdout directory
under `roboproof/runs/motion/holdout/`. Use the same Linux host and stop within
the already-approved compute/credit cap; do not add paid credit or run an
unbounded background job.
