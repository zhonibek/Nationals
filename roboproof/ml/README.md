# Deep ML readiness

This implements the **local preparation slice** of M3A/M3B/M3C and the AMD job contract. A real three/four-hidden-layer PyTorch model can train with gradients, rank new experiments, and have its predictions checked by the actual source-checked Nationals WASM controller. This is not an LLM bridge or a reinforcement-learning robot driver.

**Important:** M1 canonical browser/headless physics is still unfinished. All current ML artifacts are explicitly `EXPERIMENTAL_LEGACY_MODEL`, `canonical_ready: false`. They describe the separate reduced-order RoboProof plant, not the original browser plant or physical robot. Every entry point requires `--allow-legacy`; there is no canonical-success shortcut.

## Local workflow

Run from the repository root. The existing CPU environment is sufficient; no new dependency or account is needed. On Windows its interpreter is `roboproof/gpu/.venv/Scripts/python.exe`. The commands below use `python` to mean that interpreter (or a compatible environment with PyTorch installed using `../gpu/requirements-cpu.txt`). Use fresh output directories; frozen artifacts are never overwritten.

```powershell
node roboproof/cli.js run --count 128 --seed 5053102 --out roboproof/runs/ml-demo/corpus
node roboproof/ml/export-dataset.js --input roboproof/runs/ml-demo/corpus/report.json --out roboproof/runs/ml-demo/dataset.json --allow-legacy
python -m roboproof.ml.train prepare --input roboproof/runs/ml-demo/dataset.json --out roboproof/runs/ml-demo/package.json --seed 42 --allow-legacy
python -m roboproof.ml.train train --package roboproof/runs/ml-demo/package.json --out roboproof/runs/ml-demo/training --device cpu --epochs 20 --max-seconds 60 --allow-legacy
node roboproof/cli.js sample --count 256 --seed 5053103 --out roboproof/runs/ml-demo/pool.json
python -m roboproof.ml.acquire --model roboproof/runs/ml-demo/training/model.pt --package roboproof/runs/ml-demo/package.json --candidates roboproof/runs/ml-demo/pool.json --out roboproof/runs/ml-demo/selection.json --budget 8 --allow-legacy
node roboproof/ml/confirm.js --input roboproof/runs/ml-demo/selection.json --out roboproof/runs/ml-demo/confirmed.json --allow-legacy
```

Open the explorer using `node roboproof/server.js`. Upload **`confirmed.json`**, not `model.pt` or the dataset. It shows recorded model training, predictions versus measured categories and the selected cohort's biased pass fraction. Select a result and **Watch in simulator** to obtain and play its actual trace. That player is recorded reduced-plant motion, not canonical integration with the original browser game.

`selection.json` is a proposal, explicitly `simulation_status: NOT RUN`. Only `confirm.js` produces observed categories, and it first checks that the engine/controller hashes still match the model's evidence. Predictions never decide whether a run passes. No firmware or library gains are changed.

## Dataset and leakage boundaries

- `contract.json` versions feature order, targets, units via the linked SI scenario specification, split proportions and metric semantics. Both files' normalized source hashes are checked on export, preparation and training.
- The exporter accepts current CPU-WASM runs only. Source-checked provenance, scenario hashes, complete metrics, category/metric label consistency and exclusive SUCCESS semantics are required. Infrastructure errors are rejected, not relabeled as robot failures.
- Full demo reports export baseline rows only; candidate, search and verification holdout results are not automatically mixed in. An explicit result array has unknown sampling provenance. Historical evolutionary results without recorded mutation lineage are rejected.
- Exact renamed repeats are deduplicated; conflicting labels/lineage fail. Missing numerical-failure metrics and timeout completion times use explicit regression masks, not measured-zero labels. Telemetry hashes are retained when telemetry exists, but raw traces are not features or silently packaged for upload.
- Inputs are pre-run physical/sensor parameters, relative task geometry, heading sin/cos, dt/deadline, localization context, controller configuration and saved thresholds. IDs, seeds, hashes, final pose, failure score and post-run telemetry cannot enter feature extraction. Inactive IMU fields are zeroed; encoder measurements remain relevant even under ground-truth localization because the inner wheel loop still uses them.
- Transitive connected components join shared seeds, quantized world bins and recorded lineage roots. Paired controller configurations stay together. World-bin resolution is `1e-4` of each declared range; this is a documented approximate duplicate policy, **not a guarantee that every nearby world is joined across bin boundaries**.
- Group splits are deterministic and nonempty: approximately 60% train, 15% validation, 10% calibration, 15% final. Whole groups can change row ratios. Preprocessing is fitted on train only and rechecked at load. These splits do **not** establish unseen parameter-region or controller-build generalization; those remain separate experiments.

## Training, baselines and evidence

The default MLP has three 128-unit ReLU hidden layers, independent multi-label logits and log1p continuous-metric regression. Training uses BCE plus masked regression loss, Adam, finite-loss checks, gradient clipping, deterministic-algorithm requests, epoch/wall-time budgets and validation-based early stopping. CPU or **strict ROCm** is explicit: requesting ROCm on CPU/NVIDIA never falls back silently. Neural training uses float32; simulation precision is unchanged.

`model-card.json` records actual gradient execution, changed weights, runtime/device, epoch/loss history, dataset/split/feature/model/source hashes, engine provenance, calibration and validation metrics. Classification reports per-category recall, false negatives, false positives, precision, tie-aware average precision, Brier score and 10-bin calibration error. Undefined absent/single-class statistics remain null. Regression reports observed count and MAE in log1p space. Majority, smoothed-frequency and trained logistic baselines are retained; higher quality or search efficiency is **not assumed**.

Scalar temperature is selected using the calibration split only. Probability threshold remains 0.5. Small calibration/validation cohorts are smoke evidence, not calibrated deployment confidence. See PyTorch's [HIP detection](https://docs.pytorch.org/docs/stable/notes/hip.html), [state-dict checkpoints](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) and [reproducibility limits](https://docs.pytorch.org/docs/stable/notes/randomness.html).

`checkpoint.pt` retains optimizer, model/best model, epoch, RNG and shuffle state. Interrupted runs resume with `--resume path/to/checkpoint.pt` and the same configuration/dataset/source/runtime; completed runs are frozen. Atomic checkpoints are written at epoch boundaries, so a mid-epoch interruption restarts from the previous complete epoch. The wall-time guard is checked between training batches/phases; a provider must enforce an external hard deadline for stalled processes/GPU kernels. It is not a substitute for scheduler termination.

Training **never evaluates final**. After model/policy selection is frozen, final evaluation is a separate explicit command:

```powershell
python -m roboproof.ml.train evaluate-final --package roboproof/runs/ml-demo/package.json --model roboproof/runs/ml-demo/training/model.pt --out roboproof/runs/ml-demo/final.json --allow-legacy --unlock-final
```

The command creates a local exclusive `package.json.final-consumed.json` marker before evaluation and refuses a repeated evaluation at that package path. This is an audit guard, not access control against copying/editing files. If any final outcome influences tuning, replace the final corpus and record its reuse; do not bypass the marker and claim a fresh holdout. The readiness smoke workflow deliberately leaves final untouched.

## Learned acquisition

Acquisition excludes all existing train/validation/calibration/final group keys and exact repeats, then selects a bounded new pool with 20–30% random exploration (default 25%; rounded up, at least one). Other slots use `0.8 * max_category_probability + 0.2 * mean_predictive_entropy`. Entropy is only a heuristic, not an ensemble or proven epistemic uncertainty. Anonymous evolutionary pools are rejected until a lineage-aware adapter exists.

There is no automated retraining loop yet. Confirm selected worlds first; version and audit new labels before later retraining. Repeated matched-budget Monte Carlo/evolutionary/ML comparisons, end-to-end learning overhead, region coverage and independent-seed intervals are still required before calling this an improved search policy.

## AMD API preparation: no remote execution

```powershell
python -m roboproof.ml.train probe
python -m roboproof.ml.jobs --package roboproof/runs/ml-demo/package.json --config roboproof/ml/amd-job.example.json --out roboproof/runs/ml-demo/amd-bundle --allow-legacy
```

`probe` is read-only. The bundle contains only allowlisted ML/runtime source, scenario specification and the selected dataset package, with byte hashes, an explicit ROCm command and quotas. No entire repository, deleted/reference Pedro tree, environment files, credentials or raw telemetry is included. Bundles and outputs are ignored by Git.

`amd-job.example.json` leaves provider/endpoint/hardware unset, remote execution **disabled**, maximum spend **zero**, retries **zero**, and explicit job/concurrency/wall-time/artifact limits. A fake provider exercises submit, status, logs, cancel, timeout and hash-verified download without executing any GPU/process/network/billing operation. Fake success never counts as AMD evidence. A live provider adapter does not exist and enabling remote configuration fails closed rather than guessing an API.

When AMD access arrives, implement only its documented submit/status/log/cancel/artifact interface; verify authentication and least-privilege secret storage, GPU/OS/HIP/memory/storage capabilities and actual zero-cost entitlement or an explicitly approved spending cap. Pin the ROCm build/container then, run a small capability/parity/training probe, and only afterward scale. Nebius free inference credits do not authorize AMD compute charges. Do not paste API secrets in chat or this template.

## Tests and outstanding gates

```powershell
node --test tests/roboproof-ml.test.js tests/roboproof.test.js tests/roboproof-player.test.js tests/roboproof-server.test.js
python tests/roboproof_ml_test.py -v
```

Tests cover real WASM-derived fixture rows, feature/label checks, grouping, train-only preprocessing, corruption, actual gradient training, repeatability, interrupted resume, untouched final during training, one-shot explicit final evaluation, rare/absent-class metrics, held-out candidate exclusion, random exploration, real confirmation and fake job limits/artifact verification. Sandbox permissions may prevent Node child processes or Python temporary-directory access; these are environment failures, not passing evidence.

**Still open:** canonical shared simulator (M1/M2), canonical tensor parity and actual AMD execution (M3), independent dataset audit/region holdouts and meaningful model evaluation (M3A/B), repeated learned-search benefit and retraining (M3C), live Nebius tool investigation (M4), accepted zero-regression fix (M5), integrated release/demo (M6), supervised physical evidence (M7). ML preparation does not close those gates.
