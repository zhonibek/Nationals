# RoboProof benchmark protocol

Updated 2026-09-29. Only measured values belong here; AMD speedup is **NOT MEASURED** until a real ROCm run is retained.

## Reproduction

```powershell
node roboproof/cli.js benchmark --counts 1,100,1000,10000 --seed 42 --out roboproof/runs/cpu-benchmark.json
```

This warms the WASM baseline once, then measures complete scalar runs on seeded independently sampled scenarios, including sensor/metric work and per-run controller initialization. Every scenario lasts 10 simulated seconds with a 10 ms control tick and at most 1 ms physics substeps. Wall time, controller steps/sec and scenarios/sec are reported independently; physics substeps are not silently labeled control steps. Batch preparation and final JSON serialization are outside timing. There is no rendering during the benchmark. Other local work can affect the timing; these are single-run observations, not confidence intervals or hardware-optimized ceilings.

Measured on 2026-09-29: AMD Ryzen 3 5300U CPU, Node v24.19.0, win32/x64. Other development/test processes ran concurrently. Artifact: `roboproof/runs/cpu-benchmark.json`.

| Scenarios | Wall seconds | Scenarios/s | Control steps/s |
| ---: | ---: | ---: | ---: |
| 1 | 0.0323265 | 30.9344 | 30,934.4 |
| 100 | 1.6209435 | 61.6925 | 61,692.5 |
| 1,000 | 15.9406786 | 62.7326 | 62,732.6 |
| 10,000 | 231.564505 | 43.1845 | 43,184.5 |

These are CPU measurements, not GPU measurements. In this sandbox the benchmark's Git subprocess was denied, so `software_version` is explicitly UNVERIFIED; full simulator/controller content hashes are retained. The workspace's inspected HEAD is `f33c8392e163ac37814bb4e7cefdd5c229e89812`, and RoboProof is uncommitted. Exact uncommitted implementation identity comes from the source hashes, not that HEAD alone.

The earlier 256-world stress artifact measured 4.2618331 s (60.0680491 scenarios/s) on `cpu-wasm`. Its timing differs with workload and system load; no regression or speedup is inferred between these single observations.

## Tensor backend requirements

Measured CPU parity on 2026-09-29: PyTorch 2.8.0+cpu, float64, 510 controller lane-steps against actual WASM (maximum absolute discrepancy 1.42e-14), and 2,000 full-simulation lane-steps including all telemetry fields (maximum discrepancy 2.60e-13). Ten Python tests pass; the two actual ROCm comparisons remain skipped. CPU tensor timing and exact-repeat hashes are retained in `roboproof/gpu/validation/benchmark-cpu.json`. These comparisons validate only the exercised cases and precision, not universal numerical equivalence or GPU execution.

Use `roboproof/gpu/README.md`. Compare identical worlds and precision against the CPU oracle before performance claims. Synchronize GPU work before starting/stopping the timer. Record PyTorch and HIP/ROCm versions, reported device name, counts, duration, throughput and telemetry setting. If utilization is not obtained from a real monitoring source, report NOT MEASURED. Do not conflate a CPU-only PyTorch build, an AMD display driver, CUDA availability or a skipped test with successful AMD execution.

Benchmark 1 and 100 scalar worlds; then 100, 1,000, 10,000 and 50,000 tensor worlds as available memory allows. Large GPU counts are targets until executed. Speedup requires matched task distributions, duration, precision and timing boundaries; the scalar WASM and tensor backend have different overhead, so raw timings must remain labeled.

## Local AMD constraint

Read-only CIM inspection reports AMD Ryzen 3 5300U, `AMD Radeon(TM) Graphics`, device `0x164C`, display driver `31.0.21912.3005`. This APU is absent from the [official ROCm 7.14.1 supported APU matrix](https://rocm.docs.amd.com/en/docs-7.14.1/compatibility/compatibility-matrix.html), checked 2026-09-29. A supported AMD GPU/ROCm environment is required for the requested hardware gate. No display-driver/system changes or unsupported compatibility overrides were attempted.
