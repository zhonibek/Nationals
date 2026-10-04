# LocateAnything perception preparation

Added 2026-10-02 on `RobotAI`. **Preparation only: no model weights downloaded, live inference performed, camera connected, motion policy trained or physical robot accessed.** Nemotron's existing local text-task route is unchanged. This optional module is inside Nationals-work3 and imports the original `simulator/motion.js` task validator; it does not introduce another robot or physics engine.

## Role and current boundary

Planned chain:

```text
Consented image + target description
  -> future LocateAnything worker
  -> native image-space box/point parsing
  -> explicit target selection + reviewed field-floor point
  -> validated fixed-camera calibration
  -> shared reach-pose task proposal
  -> future separate approval/playback route
  -> original Simulator + source-checked iraLIB controller
```

LocateAnything is a visual-grounding model, not the movement learner. The [official NVIDIA model card](https://huggingface.co/nvidia/LocateAnything-3B) describes custom Transformers inference on Linux/NVIDIA GPUs and non-commercial research terms. Deployment/license approval is pending; do not assume AMD/Windows compatibility or commercial/hackathon permission. Loading its custom code will require an explicitly reviewed, pinned snapshot, not blind execution of a changing remote repository.

The module deliberately has **no inference or movement HTTP endpoint**. `GET /api/perception/status` reports readiness only, without probing providers. No environment variable can enable an unimplemented runtime. The dashboard's Vision section shows preparation and remaining gates, not a fake detection demonstration.

## Implemented contracts

`parseGrounding(input)` accepts schema version 1: `frame`, `prompt`, `answer`, `truncated: false` and `source`. Source is explicitly `fixture` (revision null) or `imported-model-output` (40-character model revision), with model ID `nvidia/LocateAnything-3B`. Imports are not independently authenticated or counted as RoboProof inference.

- Native boxes: `<box><250><250><750><750></box>`; native points: `<box><500><300></box>`. Integers are normalized to 0..1000 and scale against original image width/height. Malformed geometry, inverted/empty boxes and incomplete outputs fail closed. Other text is not an executable instruction; semantic label association and negative-token interpretation are not implemented.
- Coordinates use **pixel edges**, spanning `[0,width]` / `[0,height]`, not integer array indices. A value of 1000 is the outer edge. Crops, rotations, rectification, resizing or camera motion require a new coordinate-space/pose identity and compatible calibration; this module does not perform those image transformations.
- At most 32 detections, 32 KiB answer, 2,000 prompt characters and image dimensions up to 2560 per side. A future worker must explicitly report truncation and its bounded completion status. Confidence is `null`: no invented probability or confidence threshold.
- Frame binds ID, image SHA-256, camera/pose/profile IDs, source resolution and canonical UTC capture timestamp. The current offline importer checks metadata consistency, **not actual image bytes or EXIF**. The future capture/worker boundary must independently decode/hash the exact image, verify dimensions, disable untracked transformations and attach real provenance.

`prepareProposal(input, {now})` additionally requires `calibration`, `selection`, `start`, `goalHeadingDeg`, `deadlineSeconds`. The checked-in fixture defines the exact shape. Production callers use the default current clock; explicit `now` exists for deterministic tests, not a freshness bypass in a live route.

### Calibration and selection gates

Only a **fixed camera with rectified image coordinates observing a flat field plane** is supported. Lens distortion must be addressed before this boundary. Perspective homography alone does not handle an arbitrary moving camera, depth or raised-object geometry.

- Frame must be no older than 5 seconds and cannot be future-dated. Review must follow capture and precede proposal validation. A slow future VLM may need tracking/reacquisition or a separately validated static-scene workflow; never silently relabel old frames as fresh.
- Calibration binds camera ID, pose ID, coordinate-space ID and image size, a validity interval and a review. The homography maps to `nationals-field-inches-v1`: the existing Simulator's centered X/Y inches, heading CW from +Y. No axis, origin or unit convention is inferred from a picture.
- Calibration must supply at least four held-out measurements, declared RMS error <= 0.5 inches and maximum error <= 1 inch. These are initial software gates, **not physical safety certification**. Review records are caller-supplied; independent calibration measurement/evidence collection is still future work.
- Reject singular/ill-conditioned mappings, projective poles in the coverage rectangle, coverage outside the supported +/-60-inch reach field, calibration expiry and all extrapolation. A fixture map cannot validate a real imported output.
- User explicitly selects a detection and a pixel point reviewed as `field-floor`, bound to the frame ID/image hash. A point detection must match that point; a box must contain it. **No automatic box-center-to-goal conversion.** A human floor-plane assertion can be wrong: the future system still needs scene/depth verification and obstacle/reachability checks.

Output is `proposal_only`, simulation-only, with the original grounding, selected point, calibration snapshot, timestamp and shared task. Nothing executes. A future approval route must recheck current state/freshness and planning constraints; an offline proposal is not an authorization token or a proof that a path is safe. Physical deployment remains a separate gate.

## Try the contract without a model

From the repository root:

```powershell
node tools/prepare-perception.js --fixture roboproof/perception/fixtures/reach-point.json
node --test tests/perception.test.js
```

The synthetic fixture uses native point `(500,300)` in a 1000x1000 coordinate plane and an invented top-down map to produce the existing task `(0,0,0) -> (0,24,0)` inches/degrees. Its all-zero image hash is a **placeholder**, not a real image. `--fixture` validates at the fixture capture time, clearly labels that clock and is refused for real imported output. Normal imports use current time and will reject stale frames. The CLI reads one bounded JSON file and prints a proposal; it runs neither the model nor the simulator.

The integration test **explicitly** executes the synthetic task through `roboproof/motion.js` and the original source-checked Simulator/C++ WASM, then checks exact replay. That proves task-interface compatibility, not model accuracy, real camera calibration, learned movement or end-to-end vision navigation.

Validation on 2026-10-02: 17 perception tests pass; focused existing-engine/server/Nemotron regression totals 43 passes. Full Node/control/Simulator suite totals 102 passes, zero failures and one existing external-Java skip for unavailable compiler/runtime. The live dashboard's preparation status renders without overflow or console warnings/errors, and existing saved Nemotron evidence still restores. No LocateAnything runtime was exercised by these checks.

## Future implementation order

1. Review model/third-party licenses and intended use. Select compatible hardware and explicit resource/cost limits; pin source revision and weight hashes, review custom code, and isolate the worker. No paid/cloud request or multi-GB download is authorized by this preparation step.
2. Implement a bounded image-to-native-output worker with real model/image/runtime identities, strict timeout, output-size/completion checks, cancellation and no arbitrary model tools. Never replace unavailable inference with a fixture. Keep images local unless the user authorizes a specific remote provider.
3. Implement consented capture, rectification and independently measured camera calibration; retain held-out reprojection measurements and invalidate calibration on camera/profile changes. Evaluate ambiguity, missing targets, floor-plane selection and perception latency.
4. Evaluate real grounding/pixel-to-field errors on fixed held-out images. Keep perception scores separate from reach-task success and motion-learning rewards. Do not feed renderer/physical truth into a sensor-realistic policy as a shortcut.
5. Add reviewed, revalidated proposals and explicit approval to the existing visual Simulator/task workflow, with obstacle/reachability checks. Only then demonstrate camera-to-target motion. Physical robot access is still disabled.

**Primary next step remains the persistent-controller short-horizon action adapter, transition logging, then genuine motion-policy training and held-out improvement.** Vision is optional and does not block the initial coordinate-based training benchmark. A clicked pixel does not require a VLM; it still requires calibration and safe task planning.
