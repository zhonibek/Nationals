import os
import platform

import torch


DEVICES = ("cpu", "cuda")


def prepare_execution(name="cpu"):
    if name not in DEVICES:
        raise ValueError("Explicit CPU or NVIDIA CUDA training device required")
    identity = {"device": name, "torch": str(torch.__version__), "platform": platform.system(),
                "precision": "float32", "simulation": "Original Node/WASM Simulator and iraLIB on CPU"}
    if name == "cpu":
        return torch.device("cpu"), identity
    if torch.version.hip or not torch.version.cuda or not torch.cuda.is_available():
        raise ValueError("NVIDIA CUDA training requires an available CUDA build/device; no CPU fallback")
    if torch.cuda.device_count() != 1:
        raise ValueError("This bounded CUDA trainer requires exactly one visible GPU")
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in (":4096:8", ":16:8"):
        raise ValueError("Declared deterministic cuBLAS workspace configuration required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    identity.update({"cudaBuild": str(torch.version.cuda), "gpu": str(properties.name),
                     "capability": [properties.major, properties.minor], "memoryBytes": properties.total_memory,
                     "deviceIndex": 0, "tf32": False, "cublasWorkspace": workspace,
                     "scope": "Device visibility/runtime binding; successful network operations require actual training evidence"})
    return torch.device("cuda:0"), identity


def snapshot_cuda_rng(device):
    return torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None


def restore_cuda_rng(state, device):
    if device.type == "cuda":
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError("Saved CUDA RNG state required")
        torch.cuda.set_rng_state(state.cpu(), device)
    elif state is not None:
        raise ValueError("CPU training must not inherit a CUDA RNG state")
