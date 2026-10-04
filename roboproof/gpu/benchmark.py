"""Measured CPU or genuine AMD HIP benchmark; run as a module from the repo root."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch

from .controller import resolve_device
from .simulation import Simulation, records

ROOT = Path(__file__).resolve().parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def hardware(device):
    processor = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                processor = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    info = dict(cpu=processor, logical_cpu_count=os.cpu_count(), platform=platform.platform(),
                machine=platform.machine(), display_adapters=None, display_detection_error=None,
                display_adapter_source=None,
                measured_device="AMD ROCm" if device.type == "cuda" else "CPU")
    if sys.platform == "win32":
        try:
            result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                                     "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID | ConvertTo-Json -Compress"],
                                    capture_output=True, text=True, check=True, timeout=20,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            adapters = json.loads(result.stdout)
            info["display_adapters"] = adapters if isinstance(adapters, list) else [adapters]
            info["display_adapter_source"] = "Win32_VideoController"
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            info["display_detection_error"] = str(error)
            try:
                import winreg
                adapters = []
                key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as parent:
                    for index in range(winreg.QueryInfoKey(parent)[0]):
                        name = winreg.EnumKey(parent, index)
                        if len(name) != 4 or not name.isdigit():
                            continue
                        try:
                            with winreg.OpenKey(parent, name) as key:
                                adapters.append({field: winreg.QueryValueEx(key, value)[0] for field, value in
                                                 (("Name", "DriverDesc"), ("DriverVersion", "DriverVersion"),
                                                  ("MatchingDeviceId", "MatchingDeviceId"), ("ProviderName", "ProviderName"))})
                        except OSError:
                            continue
                if adapters:
                    info["display_adapters"] = adapters
                    info["display_adapter_source"] = "Windows installed-driver registry; not proof of active GPU or ROCm availability"
            except OSError:
                pass
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["gpu"] = dict(name=properties.name, total_memory_bytes=properties.total_memory,
                           architecture=getattr(properties, "gcnArchName", None),
                           multiprocessors=properties.multi_processor_count,
                           index=torch.cuda.current_device())
    else:
        info["gpu"] = None
    return info


def provenance():
    names = ["roboproof/core.js", "roboproof/sim.js", "roboproof/scenario-spec.json",
             "include/subsystems/control/Cascade.hpp", "simulator/native/control.cpp"]
    names += [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "roboproof/gpu").glob("*.py"))]
    sources = {name: digest((ROOT / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()) for name in names}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        revision = "UNVERIFIED"
    return dict(git_revision=revision, source_hashes=sources,
                controller_wasm_sha256=digest((ROOT / "simulator/control.wasm").read_bytes()))


def load_scenarios(path, batch_size=None):
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8-sig"))
    if isinstance(document, list):
        scenarios = document
    elif "scenarios" in document:
        scenarios = document["scenarios"]
    elif "scenario" in document:
        scenarios = [document["scenario"]]
    else:
        scenarios = [document]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Expected a nonempty scenario array, {scenarios: [...]}, or single scenario")
    original_count = len(scenarios)
    if batch_size is not None:
        scenarios = [scenarios[index % original_count] for index in range(batch_size)]
    return scenarios, dict(input_file=str(path.resolve()), input_sha256=digest(raw),
                           input_count=original_count, batch_size=len(scenarios),
                           cycling_input=batch_size is not None and batch_size > original_count)


def benchmark(scenarios, device_name="cpu", repeats=3, warmup=1, threads=1):
    device = resolve_device(device_name)
    if repeats < 1 or warmup < 0 or threads < 1:
        raise ValueError("repeats and threads must be positive; warmup must be nonnegative")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    timings, hashes, setup_times = [], [], []
    final_records = None
    for repeat in range(warmup + repeats):
        synchronize(device)
        setup_start = time.perf_counter()
        simulation = Simulation(scenarios, device=device_name)
        synchronize(device)
        setup_seconds = time.perf_counter() - setup_start
        synchronize(device)
        start = time.perf_counter()
        result = simulation.run(telemetry=False)
        synchronize(device)
        elapsed = time.perf_counter() - start
        if repeat >= warmup:
            timings.append(elapsed)
            setup_times.append(setup_seconds)
            final_records = records(result)
            hashes.append(digest(json.dumps(final_records, sort_keys=True, allow_nan=False).encode()))
    total_steps = sum(record["metrics"]["steps"] for record in final_records)
    median = statistics.median(timings)
    return dict(schema_version=1, measured_at=datetime.now(timezone.utc).isoformat(),
                backend="torch-rocm" if device_name == "rocm" else "torch-cpu", gpu_measured=device_name == "rocm",
                hardware=hardware(device), versions=dict(python=platform.python_version(), torch=torch.__version__,
                hip=torch.version.hip, cuda=torch.version.cuda), provenance=provenance(), dtype="float64",
                timing=dict(clock="time.perf_counter", synchronization="torch.cuda.synchronize before and after each timed region" if device.type == "cuda" else "CPU operations complete synchronously",
                            scope="Simulation.run without telemetry, includes DARE solves and 1 ms physics; excludes setup, transfer of final results to CPU, serialization and hardware query",
                            repeats=repeats, warmup_runs=warmup, seconds=timings, median_seconds=median,
                            setup_seconds=setup_times, simulations_per_second=len(scenarios) / median,
                            control_steps_per_second=total_steps / median,
                            simulated_seconds_per_wall_second=total_steps * scenarios[0]["task"]["dt"] / median),
                workload=dict(batch_size=len(scenarios), dt=scenarios[0]["task"]["dt"],
                              duration=scenarios[0]["task"]["duration"], executed_control_steps=total_steps),
                determinism=dict(algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
                                 cpu_threads=torch.get_num_threads(), rng="uint32 LCG 1664525/1013904223; Box-Muller; separate encoder/IMU streams",
                                 scenario_seeds=[scenario["random_seed"] for scenario in scenarios],
                                 result_sha256=hashes, repeated_results_exact=len(set(hashes)) == 1 if repeats > 1 else None,
                                 comparison_scope="final truth, estimate, metrics, categories and failure score; not full telemetry",
                                 cross_backend="float64 tolerance required; no cross-hardware bitwise guarantee"),
                limitations=["Engineering model, not calibrated hardware validation or a safety certificate",
                             "Measured under current system load; no exclusive hardware reservation or power-state control",
                             "Tensor batch requires a common dt and duration; batch initialization and serialization use Python",
                             "DARE is capped at 600 iterations and uses host convergence checks; small GPU batches may be slower",
                             "No GPU throughput is inferred from CPU measurements or the presence of an AMD display adapter",
                             "This benchmark alone does not establish JS/WASM or CPU/ROCm parity; run the parity tests",
                             "Numerically failed lanes stop metric accounting but remain in the tensor workload"],
                results=final_records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "rocm"), default="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        resolve_device(args.device)
        if args.batch_size is not None and args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        scenarios, source = load_scenarios(args.scenarios, args.batch_size)
        report = benchmark(scenarios, args.device, args.repeats, args.warmup, args.threads)
        report["input"] = source
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
        print(text, end="")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"RoboProof benchmark error: {error}\n")


if __name__ == "__main__":
    main()
