# CPU validation evidence

These small artifacts are intentionally versionable; they are measured evidence,
not generated golden expectations for the parity tests:

- `scenarios.json`: eight scenarios exported by the actual Node CLI with
  `sample --count 8 --seed 42`; each simulates 10 seconds at a 10 ms control tick.
- `benchmark-cpu.json`: real CPU timing, source/input hashes, hardware metadata,
  two measured repeats after one warmup, exact-repeat checks and final metrics.
  No GPU run is represented. Timings describe current system load, not an
  exclusive-core or controlled-power benchmark. This eager tensor implementation
  is correctness-first; this report does not assert a speedup over Node.

The local `.venv`, Python bytecode and caches are ignored by `gpu/.gitignore`.
Large sweeps should go into the existing ignored `roboproof/runs` directory,
not into this evidence folder.

## Completed checks

Runtime: Python 3.12.14, PyTorch **2.8.0+cpu**, float64, one CPU thread.
Full suite: **12 tests in 65.020 seconds; 10 passed, 2 skipped**. Both skips
explicitly report that a genuine AMD HIP/ROCm device is unavailable.
Repeated on 2026-09-29: 12 tests in 64.465 seconds, again 10 passed and the same
2 hardware-dependent tests skipped. Test-suite wall time is not a throughput measurement.

```powershell
& './roboproof/gpu/.venv/Scripts/python.exe' tests/roboproof_gpu_test.py
```

- Actual WASM controller: 510 lane-steps, maximum absolute discrepancy
  `1.42e-14` across voltages and wheel targets.
- Actual scalar JS/WASM simulation: 2,000 lane-steps, every telemetry field,
  nominal/stressed/ground-truth/wrap scenarios; maximum absolute discrepancy
  `2.60e-13`. Metrics, classifications and final states also match.
- Replay and duplicate-lane isolation pass exactly. The real benchmark's
  final-result hashes separately verify repeated-run determinism.
- The suite covers the 600-iteration DARE cap, cache refresh/reset, production
  inversion cutoff, wheel derivative filter, sensor faults and NVIDIA rejection.
- The actual CLI `--device rocm` request returns status **2** with
  `AMD ROCm requested, but torch.version.hip and an available HIP device are required; no GPU measurement was performed`.
  It emits no JSON GPU report. `torch.version.hip` and `torch.version.cuda` are
  both null in the installed CPU build.

The Windows display driver identifies AMD Radeon(TM) Graphics,
31.0.21912.3005. Registry driver metadata does not establish ROCm support.
Use a supported, genuine HIP runtime/device and rerun the two currently skipped
tests before claiming GPU parity or measuring GPU throughput.
