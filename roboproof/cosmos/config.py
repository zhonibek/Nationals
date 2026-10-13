"""Dependency-free, bounded configuration; no device or network work on import."""

from dataclasses import dataclass, field
import math
from pathlib import Path
import re

from .errors import CosmosError


MODEL_ID = "nvidia/Cosmos-Reason2-2B"
MODEL_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"


@dataclass(frozen=True)
class Limits:
    max_upload_bytes: int = 32 * 1024 * 1024
    max_duration_seconds: float = 30.0
    sampling_fps: float = 4.0
    max_frames: int = 32
    max_native_pixels: int = 8_294_400
    max_frame_pixels: int = 262_144
    max_decoded_frames: int = 512
    max_decoded_pixels: int = 512_000_000
    max_decode_seconds: float = 15.0
    max_instruction_chars: int = 4000
    max_input_tokens: int = 8192
    max_output_tokens: int = 4096

    def __post_init__(self):
        integer_bounds = {
            "max_upload_bytes": (1, 64 * 1024 * 1024),
            "max_frames": (2, 64),
            "max_native_pixels": (1024, 16_777_216),
            "max_frame_pixels": (16_384, 524_288),
            "max_decoded_frames": (2, 2048),
            "max_decoded_pixels": (1024, 1_024_000_000),
            "max_instruction_chars": (1, 8000),
            "max_input_tokens": (256, 8192),
            "max_output_tokens": (1, 4096),
        }
        for name, (minimum, maximum) in integer_bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise CosmosError("configuration")
        for name, maximum in (("max_duration_seconds", 60), ("sampling_fps", 4), ("max_decode_seconds", 60)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= maximum:
                raise CosmosError("configuration")
        if self.max_frames % 2 or self.max_frame_pixels > self.max_native_pixels:
            raise CosmosError("configuration")


@dataclass(frozen=True)
class RuntimeConfig:
    media_roots: tuple[Path, ...]
    revision: str = MODEL_REVISION
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    allow_model_download: bool = False
    min_total_gpu_gib: float = 24.0
    min_free_gpu_gib: float = 8.0
    limits: Limits = field(default_factory=Limits)

    def __post_init__(self):
        if not isinstance(self.media_roots, tuple) or not self.media_roots or not isinstance(self.limits, Limits):
            raise CosmosError("configuration")
        if not isinstance(self.device, str) or not re.fullmatch(r"cuda(?::[0-9]{1,2})?", self.device) or self.dtype not in ("bfloat16", "float16"):
            raise CosmosError("configuration")
        if type(self.allow_model_download) is not bool:
            raise CosmosError("configuration")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", self.revision) or ".." in self.revision:
            raise CosmosError("configuration")
        for value in (self.min_total_gpu_gib, self.min_free_gpu_gib):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 8 <= value <= 1024:
                raise CosmosError("configuration")
        try:
            roots = tuple(Path(root).expanduser().resolve(strict=True) for root in self.media_roots)
            if any(not root.is_dir() or str(root).startswith(("\\\\", "//")) for root in roots):
                raise CosmosError("configuration")
        except (OSError, TypeError, ValueError, RuntimeError):
            raise CosmosError("configuration") from None
        object.__setattr__(self, "media_roots", roots)
