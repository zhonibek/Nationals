"""Bounded, metadata-only ROCm diagnostics, without training or tensor allocation.

Run ``python -B -m roboproof.gpu.preflight`` from the repository root. Exit codes
are 0 for visible HIP capability, 2 for unavailable capability, and 1 for probe
errors or timeout. Capability is not operator, parity, or performance validation.
The legacy ML probe's ``rocm_ready`` and runtime metadata keys are retained;
``metadata.device`` describes the reported runtime, not an executed workload.
"""

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


DEFAULT_TIMEOUT_SECONDS = 15.0
MAX_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_DEVICES = 8
MAX_DEVICES = 16
MAX_REPORT_BYTES = 65536
MAX_TEXT_LENGTH = 256


def _text(value):
    return None if value is None else str(value).strip()[:MAX_TEXT_LENGTH]


def _report():
    return dict(
        schema_version=1, status="unverified", reason="probe_not_completed", rocm_ready=False,
        metadata=dict(device=None, torch=None, hip=None, cuda=None,
                      python=platform.python_version(), platform=sys.platform,
                      machine=None, gpu=None, gpu_memory_bytes=None,
                      gpu_architecture=None, cpu_threads=None),
        cpu=dict(status="unverified", logical_cpu_count=None, processor=None,
                 operations_tested=False),
        rocm=dict(hip_build=None, device_available=None, device_count=None,
                  devices=[], devices_truncated=False),
        errors=[],
        limitations=[
            "Metadata only: no tensors, kernels, training, simulation or performance measurements.",
            "CPU status records PyTorch import, not a successful CPU operation.",
            "Visible HIP devices do not establish operator support, parity or official GPU/OS compatibility.",
            "Memory is total device capacity, not free memory or a workload-fit guarantee.",
        ],
    )


def _failure(report, status, reason, stage=None, error=None):
    report.update(status=status, reason=reason, rocm_ready=False)
    if error is not None:
        report["errors"].append(dict(stage=stage, type=_text(type(error).__name__),
                                     message=_text(error)))
    return report


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def _collect(max_devices):
    report = _report()
    stage = "host_metadata"
    try:
        report["metadata"].update(platform=_text(platform.platform()), machine=_text(platform.machine()))
        report["cpu"].update(logical_cpu_count=os.cpu_count(), processor=_text(platform.processor()))
        stage = "torch_import"
        torch = importlib.import_module("torch")
        report["cpu"]["status"] = "torch_imported"
        report["metadata"].update(device="cpu", torch=_text(torch.__version__))
        stage = "runtime_metadata"
        for name, value in (("hip", torch.version.hip), ("cuda", torch.version.cuda)):
            if value is not None and not isinstance(value, str):
                raise ValueError(name + " version must be a string or null")
        hip = _text(torch.version.hip)
        cuda = _text(torch.version.cuda)
        report["metadata"].update(hip=hip, cuda=cuda)
        report["rocm"]["hip_build"] = bool(hip)
        report["metadata"]["cpu_threads"] = _positive_integer(torch.get_num_threads(), "cpu_threads")
        if not hip:
            return _failure(report, "blocked", "non_hip_build" if cuda else "cpu_only_build")
        if cuda:
            return _failure(report, "blocked", "ambiguous_hip_cuda_build")
        stage = "device_availability"
        available = torch.cuda.is_available()
        if type(available) is not bool:
            raise ValueError("device availability must be a boolean")
        report["rocm"]["device_available"] = available
        if not available:
            return _failure(report, "blocked", "hip_device_unavailable")
        stage = "device_count"
        count = torch.cuda.device_count()
        if type(count) is not int or count < 0:
            raise ValueError("device_count must be a nonnegative integer")
        report["rocm"].update(device_count=count, devices_truncated=count > max_devices)
        if count == 0:
            return _failure(report, "blocked", "no_visible_hip_devices")
        for index in range(min(count, max_devices)):
            stage = "device_properties:" + str(index)
            properties = torch.cuda.get_device_properties(index)
            if not isinstance(properties.name, str):
                raise ValueError("device name must be a string")
            name = _text(properties.name)
            if not name:
                raise ValueError("device name must not be empty")
            memory = _positive_integer(properties.total_memory, "total_memory_bytes")
            report["rocm"]["devices"].append(dict(
                index=index, name=name, total_memory_bytes=memory,
                architecture=_text(getattr(properties, "gcnArchName", None)),
            ))
        first = report["rocm"]["devices"][0]
        report["metadata"].update(device="rocm", gpu=first["name"],
                                  gpu_memory_bytes=first["total_memory_bytes"],
                                  gpu_architecture=first["architecture"])
        report.update(status="ready", reason="hip_devices_visible", rocm_ready=True)
        return report
    except ModuleNotFoundError as error:
        if stage == "torch_import" and error.name == "torch":
            report["cpu"]["status"] = "pytorch_missing"
            return _failure(report, "blocked", "pytorch_missing", stage, error)
        if stage == "torch_import":
            report["cpu"]["status"] = "import_error"
        return _failure(report, "error", "probe_failed", stage, error)
    except Exception as error:
        if stage == "torch_import":
            report["cpu"]["status"] = "import_error"
        return _failure(report, "error", "probe_failed", stage, error)


def _validate_report(report, max_devices):
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("invalid probe schema")
    if report.get("status") not in ("ready", "blocked", "error"):
        raise ValueError("invalid probe status")
    if type(report.get("rocm_ready")) is not bool or report["rocm_ready"] != (report["status"] == "ready"):
        raise ValueError("inconsistent ROCm readiness")
    for name in ("metadata", "cpu", "rocm"):
        if not isinstance(report.get(name), dict):
            raise ValueError("missing probe metadata: " + name)
    if not isinstance(report.get("errors"), list) or not isinstance(report.get("limitations"), list):
        raise ValueError("missing probe diagnostics")
    if report["rocm_ready"]:
        metadata = report["metadata"]
        rocm = report["rocm"]
        hip = metadata.get("hip")
        if not isinstance(hip, str) or not hip.strip() or metadata.get("cuda") or metadata.get("device") != "rocm":
            raise ValueError("ready report requires a genuine HIP build")
        if rocm.get("hip_build") is not True or rocm.get("device_available") is not True:
            raise ValueError("ready report requires an available HIP device")
        count = _positive_integer(rocm.get("device_count"), "device_count")
        devices = rocm.get("devices")
        if not isinstance(devices, list) or len(devices) != min(count, max_devices):
            raise ValueError("incomplete device metadata")
        for index, device in enumerate(devices):
            if not isinstance(device, dict) or type(device.get("index")) is not int or device["index"] != index:
                raise ValueError("invalid device index")
            name = device.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("invalid device metadata")
            _positive_integer(device.get("total_memory_bytes"), "total_memory_bytes")
    return report


def probe(timeout_seconds=DEFAULT_TIMEOUT_SECONDS, max_devices=DEFAULT_MAX_DEVICES):
    """Inspect the current interpreter in one killable, bytecode-disabled child.

    The timeout covers child import and all runtime queries. No ROCm request is
    replaced with NVIDIA CUDA or a CPU workload. Missing packages are reported,
    never installed. The caller's Torch state and thread settings are untouched.
    """
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise ValueError("timeout_seconds must be a finite number between 0.1 and 60")
    if not math.isfinite(timeout_seconds) or not 0.1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise ValueError("timeout_seconds must be a finite number between 0.1 and 60")
    _positive_integer(max_devices, "max_devices")
    if max_devices > MAX_DEVICES:
        raise ValueError("max_devices must not exceed " + str(MAX_DEVICES))
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--_worker", str(max_devices)]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=timeout_seconds, check=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode != 0:
            raise RuntimeError("probe child exited with status " + str(result.returncode))
        if len(result.stdout) > MAX_REPORT_BYTES:
            raise ValueError("probe output exceeds byte limit")
        return _validate_report(json.loads(result.stdout), max_devices)
    except subprocess.TimeoutExpired as error:
        return _failure(_report(), "timeout", "probe_timeout", "subprocess", error)
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
        return _failure(_report(), "error", "probe_failed", "subprocess", error)


def _worker(max_devices):
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as output:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), sys.stdout.fileno())
            os.dup2(sink.fileno(), sys.stderr.fileno())
            report = _collect(max_devices)
            encoded = json.dumps(report, allow_nan=False)
            if len(encoded.encode("utf-8")) + 1 > MAX_REPORT_BYTES:
                encoded = json.dumps(_failure(_report(), "error", "report_byte_limit"), allow_nan=False)
            output.write(encoded + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-devices", type=int, default=DEFAULT_MAX_DEVICES)
    options = parser.parse_args(argv)
    try:
        report = probe(options.timeout_seconds, options.max_devices)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, allow_nan=False))
    return 0 if report["status"] == "ready" else 2 if report["status"] == "blocked" else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--_worker":
        worker_limit = _positive_integer(int(sys.argv[2]), "max_devices")
        if worker_limit > MAX_DEVICES:
            raise SystemExit(2)
        _worker(worker_limit)
    else:
        raise SystemExit(main())
