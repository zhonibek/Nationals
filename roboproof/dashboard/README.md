# Failure explorer

The **Vision** section reports LocateAnything preparation only: native parsing, reviewed calibration gates and the shared task contract. Its status request never invokes a model. No model weights, camera connection or perception execution endpoint are enabled. See [future perception integration](../perception/README.md).

The RobotAI panel now supports local Nemotron: task -> reviewed proposal -> original Simulator/iraLIB execution -> measured result and exact replay. **Restore saved session** loads disk-backed model sessions; this does not persist older scalar batches or train a motion policy. The original 2D/3D Simulator is served from the same API under `/simulator/` and loads the selected saved task without automatically starting it. See [local model setup](../NEMOTRON.md).

Run `node roboproof/server.js` from the repository root and open http://127.0.0.1:8766. No build, package install, CDN or account is needed. The local worker API runs the real source-checked controller; the Run button is not a prerecorded animation. Counts are bounded to 1,000 for interactive use.

You can also open `index.html` directly and upload a full `report.json`. Embedded counterexample telemetry works offline; other scenarios require the local Replay endpoint. Replay displays whether the newly measured metrics match the recorded metrics. A replay of old source/runtime data may legitimately differ and is not silently labeled reproduced.

The baseline, adversarial, holdout and candidate cohorts remain separate. Candidate rejection and regression counts are prominent. The report-level diagnosis stays attached to its recorded counterexample; selecting another scenario never reassigns that hypothesis. Default encoder-only worlds label the IMU inactive.

Graphs contain actual telemetry only. Planned/truth/estimate use distinct colors and line styles. Spatial paths use equal X/Y scale. The parameter map shows observations, not an interpolated probability surface. Above 4,000 points it explicitly subsamples for display; the paginated table and selector retain all results. Trace rendering subsamples to about 800 points while the expandable telemetry table retains every sample. JSON upload is bounded to 64 MiB; text is rendered as text, not HTML.

## Browser validation

On 2026-09-30 the explorer also passed local checks for a genuinely trained CPU neural model's 8 confirmed selected experiments: training/device/epoch metadata, scenario-specific predicted versus measured categories, exact metric replay and 1,000-sample robot playback. ML-selected cohorts are labeled biased rather than baseline population estimates. The user's successful scenario JSON imported directly, showed PASS and played the actual recorded motion; the ML panel correctly hid for this non-ML input. No console errors/warnings were captured. Hash text and the new ML table wrap for the narrow in-app panel. See [ML workflow](../ml/README.md); this does not establish original-browser/headless physics parity.

Checked in the local in-app browser on 2026-09-29:

- The full retained report loads 960 results across separate cohorts and prominently reports REJECTED.
- Baseline `mc-42-0` initially has no stored trace; Replay generates 1,000 samples and reports exact metric equality with the recorded result.
- Running count 2, seed 42 produces a real CPU batch with 1/2 passes.
- Loading the saved `report.json` through the file input restores the full report and its recorded adversarial counterexample.
- No browser console errors or warnings were recorded during these checks. The narrow local panel renders its controls without horizontal page overflow.

These are manual UI checks, not a cross-browser or assistive-technology certification. API validation and model-protocol checks also have automated Node tests.
