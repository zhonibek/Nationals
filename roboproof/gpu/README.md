# RoboProof tensor backend

Float64 PyTorch translation of `include/subsystems/control/Cascade.hpp` and
`roboproof/sim.js`. CPU is the default, including on machines with an AMD display
adapter. No GPU performance is inferred from CPU performance.

## Local CPU environment

From the repository root in PowerShell:

```powershell
& 'C:\Users\kassi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv --system-site-packages roboproof/gpu/.venv
& './roboproof/gpu/.venv/Scripts/python.exe' -m pip install -r roboproof/gpu/requirements-cpu.txt
& './roboproof/gpu/.venv/Scripts/python.exe' tests/roboproof_gpu_test.py
```

The local environment is ignored by Git. The CPU wheel installed for this work
was `torch 2.8.0+cpu` (a 619.4 MB download on Windows); reuse this environment and
the pip cache rather than downloading it repeatedly. NumPy alone is not enough.
Tests explicitly skip if PyTorch is missing; that is **not** a successful CPU
validation. Node must be on PATH for the actual WASM/JS oracle.

## Benchmark an exported scenario file

```powershell
node roboproof/cli.js sample --count 8 --seed 42 --out roboproof/gpu/scenarios.json
& './roboproof/gpu/.venv/Scripts/python.exe' -m roboproof.gpu.benchmark --scenarios roboproof/gpu/scenarios.json --device cpu --warmup 1 --repeats 3 --out roboproof/gpu/benchmark-cpu.json
```

Accepted inputs: the Node CLI's `{scenarios: [...]}`, a scenario array, a single
scenario, or a Node result with a `scenario` member. `--batch-size N` explicitly
cycles/truncates the supplied scenarios; reports disclose cycling. All lanes in
one batch must share `task.dt` and `task.duration`; group other cohorts separately.
Different environments, seeds, goals, localization modes and controller geometry
are supported within a batch. `--threads` defaults to 1.

Reports include actual processor and OS identity, detected Windows display
adapters/driver versions, Python/PyTorch/HIP/CUDA versions, source and input hashes,
raw wall-clock timings, throughput, warmup count, setup timings, deterministic
settings, repeat result hashes, metrics and limitations. Timed stepping includes
DARE solves, sensors, estimation, metrics and 1 ms physics substeps. Initialization,
result downloads/serialization and hardware queries are excluded and identified.
HIP timing synchronizes before and after each timed region. CPU operations are
synchronous. Throughput is not a claim that this eager implementation is faster
than Node or that its physical model is calibrated.

If Windows CIM queries are unavailable, display metadata falls back to installed
driver registry entries and explicitly identifies that source. Installed drivers
are not proof of an active GPU or HIP compatibility. CPU identity is read from
the processor registry, with the platform processor string as fallback.

## Genuine AMD ROCm only

```powershell
python -m roboproof.gpu.benchmark --scenarios roboproof/gpu/scenarios.json --device rocm --out roboproof/gpu/benchmark-rocm.json
```

Use this command only in a separately installed, supported AMD HIP/ROCm PyTorch
environment. The CPU requirements file deliberately does **not** install ROCm.
`--device rocm` requires both `torch.version.hip` and
`torch.cuda.is_available()`. The `cuda` API namespace is PyTorch's HIP interface;
a CUDA/NVIDIA-only build is explicitly rejected, even if its GPU is available.
There is no silent fallback after an explicit ROCm request. The benchmark exits
with status 2 and a diagnostic rather than fabricating a GPU result.

The detected `AMD Radeon(TM) Graphics`, driver `31.0.21912.3005`, is not evidence
of ROCm compatibility. Exact support has not been established, and the installed
CPU runtime has no HIP support. No GPU parity or GPU throughput has been measured
in this environment. The tests contain separate real-ROCm controller and
simulation parity checks and give explicit skip reasons when HIP is absent.

## Fidelity and validation

- Controller: 600-iteration DARE cap, relative convergence test, partial-pivot
  inversion cutoff, per-lane convergence/cache state and exact cache thresholds;
  uniform wheel desaturation/ramping, PI anti-windup, smooth static/velocity/
  acceleration feedforward and the 30 ms derivative filter. `configure()` resets
  the cache as the WASM configuration entry point does.
- Simulator: batched implicit motor/contact dynamics, friction-limited traction,
  separate truth/odometry, sample-and-hold sensors with rounded-up latency/period,
  seeded uint32 LCG plus Box-Muller streams, quantization, dropout, command delay,
  quintic reference, failure classification and metrics. The JSON spec is read,
  not duplicated as an independent set of environmental defaults.
- Python loops iterate simulation time, physics substeps, fixed matrix dimensions,
  fixed wheel channels or field names—not environments during stepping. Input
  validation/packing and final record serialization may iterate scenarios.
- DARE convergence uses a host-visible batch check and can synchronize a HIP
  device each iteration. Small GPU batches may therefore be slower. This is an
  eager, correctness-first implementation, not a fused GPU kernel.
- `wasm_fixture.cjs` verifies the actual WASM binary and all source hashes against
  its build manifest, then calls the real C++ exports. It does not replace the
  controller with a JavaScript approximation.
- Tests compare 510 controller lane-steps to WASM, including cache refreshes,
  saturation, invalid inputs, resets and failure at the iteration cap. The
  nonzero derivative gain is checked analytically because the WASM ABI exposes
  only its production zero-D configuration.
- Full scalar comparison covers 2,000 simulation lane-steps and every telemetry
  field, metrics, categories and final state across nominal, adverse sensors/
  traction, ground-truth baseline and heading-wrap scenarios. Extra checks cover
  a different timestep, same-backend replay and lane isolation.
- Initial measured CPU maximum absolute discrepancies were `1.42e-14` for the
  controller and `2.60e-13` for full simulation telemetry, below the `2e-8`
  simulation tolerance. Re-run after changes to either backend.
- Same-runtime repeated results are checked exactly; cross-device transcendental
  functions and matrix arithmetic require tolerances. This is not a bitwise
  reproducibility promise across platforms. Numerically failed lanes stop metric
  accounting but remain in the batched workload; serialized final truth is null.

Tests: `tests/roboproof_gpu_test.py`. Benchmark entry point:
`python -m roboproof.gpu.benchmark --help`.
