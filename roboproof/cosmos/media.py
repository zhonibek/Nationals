"""Local-only admission and bounded decoding before any model load."""

from dataclasses import dataclass, replace
import hashlib
import importlib
import io
import math
import os
from pathlib import Path
import re
import stat
import time
import warnings

from .config import Limits, RuntimeConfig
from .errors import CosmosError, sanitized_error


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
VIDEO_EXTENSIONS = frozenset({".mp4"})


@dataclass(frozen=True)
class PreparedMedia:
    kind: str
    frames: tuple
    timestamps_seconds: tuple[float, ...] = ()
    duration_seconds: float | None = None
    source_fps: float | None = None
    source_frame_count: int | None = None
    input_sha256: str | None = None

    def provenance(self) -> dict:
        return {
            "kind": self.kind,
            "input_sha256": self.input_sha256,
            "sampled_frames": len(self.frames),
            "sampled_timestamps_seconds": list(self.timestamps_seconds),
            "duration_seconds": self.duration_seconds,
            "source_fps": self.source_fps,
            "processor_timestamp_basis": "source average FPS, rounded frame indices" if self.kind == "video" else None,
        }


def validate_instruction(instruction: str, limits: Limits) -> str:
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > limits.max_instruction_chars:
        raise CosmosError("instruction")
    if any(ord(character) < 32 and character not in "\n\r\t" for character in instruction) or re.search(r"<\|[^\n]*?\|>", instruction):
        raise CosmosError("instruction")
    return instruction.strip()


def validate_output_tokens(value: int, limits: Limits) -> int:
    if type(value) is not int or not 1 <= value <= limits.max_output_tokens:
        raise CosmosError("tokens")
    return value


def guard_media_path(value: str | Path, config: RuntimeConfig) -> Path:
    if not isinstance(value, (str, Path)):
        raise CosmosError("path")
    raw = str(value)
    if not raw or "\x00" in raw or raw.startswith(("\\\\", "//")) or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw) and not re.match(r"^[A-Za-z]:[\\/]", raw):
        raise CosmosError("path")
    path = Path(raw)
    if ".." in path.parts or ":" in raw[2:]:
        raise CosmosError("path")
    try:
        absolute = path.absolute()
        if any(
            part.is_symlink()
            or getattr(part, "is_junction", lambda: False)()
            or getattr(part.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            for part in (absolute, *absolute.parents)
        ):
            raise CosmosError("path")
        resolved = absolute.resolve(strict=True)
        if not any(resolved.is_relative_to(root) for root in config.media_roots) or not stat.S_ISREG(resolved.stat().st_mode):
            raise CosmosError("path")
    except (OSError, ValueError, RuntimeError):
        raise CosmosError("path") from None
    if resolved.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        raise CosmosError("media_type")
    return resolved


def read_media_bytes(value: str | Path, config: RuntimeConfig) -> tuple[Path, bytes]:
    path = guard_media_path(value, config)
    descriptor = None
    try:
        before = path.stat()
        if not 0 < before.st_size <= config.limits.max_upload_bytes:
            raise CosmosError("upload")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as media_file:
            descriptor = None
            opened = os.fstat(media_file.fileno())
            if not stat.S_ISREG(opened.st_mode) or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise CosmosError("path")
            content = media_file.read(config.limits.max_upload_bytes + 1)
            after = os.fstat(media_file.fileno())
        if guard_media_path(path, config) != path or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise CosmosError("upload")
        if not 0 < len(content) <= config.limits.max_upload_bytes or len(content) != before.st_size:
            raise CosmosError("upload")
        return path, content
    except CosmosError:
        raise
    except (OSError, ValueError):
        raise CosmosError("path") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def sample_times(duration_seconds: float, limits: Limits) -> tuple[float, ...]:
    if not math.isfinite(duration_seconds) or not 0 < duration_seconds <= limits.max_duration_seconds:
        raise CosmosError("duration")
    count = min(limits.max_frames, max(2, math.ceil(duration_seconds * limits.sampling_fps)))
    count = min(limits.max_frames, count + count % 2)
    return tuple(index * duration_seconds / count for index in range(count))


def bounded_dimensions(width: int, height: int, limits: Limits) -> tuple[int, int]:
    if width < 32 or height < 32 or width * height > limits.max_native_pixels or max(width, height) / min(width, height) > 32:
        raise CosmosError("pixels")
    scale = min(1.0, math.sqrt(limits.max_frame_pixels / (width * height)))
    result = (max(32, math.floor(width * scale / 32) * 32), max(32, math.floor(height * scale / 32) * 32))
    if result[0] * result[1] > limits.max_frame_pixels:
        raise CosmosError("pixels")
    return result


def _prepare_image(content: bytes, limits: Limits) -> PreparedMedia:
    image_module = importlib.import_module("PIL.Image")
    image_ops = importlib.import_module("PIL.ImageOps")
    with warnings.catch_warnings():
        warnings.simplefilter("error", image_module.DecompressionBombWarning)
        with image_module.open(io.BytesIO(content)) as source:
            if source.format not in ("JPEG", "PNG") or getattr(source, "n_frames", 1) != 1:
                raise CosmosError("media")
            bounded_dimensions(*source.size, limits)
            source.load()
            oriented = image_ops.exif_transpose(source)
            dimensions = bounded_dimensions(*oriented.size, limits)
            image = oriented.convert("RGB").resize(dimensions, image_module.Resampling.LANCZOS)
    return PreparedMedia("image", (image,))


def _prepare_video(content: bytes, limits: Limits) -> PreparedMedia:
    av_module = importlib.import_module("av")
    frames = []
    timestamps = []
    decoded_frames = 0
    decoded_pixels = 0
    deadline = time.monotonic() + limits.max_decode_seconds
    options = {"enable_drefs": "0", "use_absolute_path": "0", "protocol_whitelist": "pipe"}
    with av_module.open(io.BytesIO(content), mode="r", format="mov", options=options) as container:
        if len(container.streams.video) != 1:
            raise CosmosError("media")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        dimensions = bounded_dimensions(stream.codec_context.width, stream.codec_context.height, limits)
        if stream.duration is None or stream.time_base is None or not stream.average_rate:
            raise CosmosError("duration")
        duration = float(stream.duration * stream.time_base)
        if container.duration is not None:
            container_duration = float(container.duration / av_module.time_base)
            sample_times(container_duration, limits)
        targets = sample_times(duration, limits)
        source_fps = float(stream.average_rate)
        if not math.isfinite(source_fps) or not 0 < source_fps <= 120:
            raise CosmosError("media")
        if container.duration is not None and duration > container_duration + max(0.25, 2 / source_fps):
            raise CosmosError("duration")
        start_pts = stream.start_time or 0
        start_seconds = float(start_pts * stream.time_base)
        for target in targets:
            if time.monotonic() > deadline:
                raise CosmosError("decode_limit")
            container.seek(start_pts + int(target / float(stream.time_base)), stream=stream, backward=True, any_frame=False)
            selected = None
            selected_time = None
            for frame in container.decode(stream):
                decoded_frames += 1
                decoded_pixels += frame.width * frame.height
                if decoded_frames > limits.max_decoded_frames or time.monotonic() > deadline:
                    raise CosmosError("decode_limit")
                if decoded_pixels > limits.max_decoded_pixels:
                    raise CosmosError("pixels")
                bounded_dimensions(frame.width, frame.height, limits)
                if frame.time is None:
                    raise CosmosError("duration")
                timestamp = float(frame.time) - start_seconds
                if not math.isfinite(timestamp) or timestamp < -0.05 or timestamp > duration + 1 / source_fps or timestamp > limits.max_duration_seconds:
                    raise CosmosError("duration")
                selected = frame
                selected_time = max(0.0, timestamp)
                if timestamp + 0.000001 >= target:
                    break
            if selected is None or selected_time is None or abs(selected_time - target) > max(0.25, 2 / source_fps):
                raise CosmosError("media")
            if timestamps and selected_time < timestamps[-1]:
                raise CosmosError("duration")
            frames.append(selected.reformat(width=dimensions[0], height=dimensions[1], format="rgb24").to_image())
            timestamps.append(selected_time)
        source_count = stream.frames or max(1, math.ceil(duration * source_fps))
    return PreparedMedia("video", tuple(frames), tuple(timestamps), duration, source_fps, source_count)


def prepare_media(value: str | Path, config: RuntimeConfig) -> PreparedMedia:
    path, content = read_media_bytes(value, config)
    try:
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            prepared = _prepare_image(content, config.limits)
        else:
            prepared = _prepare_video(content, config.limits)
        return replace(prepared, input_sha256=hashlib.sha256(content).hexdigest())
    except Exception as error:
        raise sanitized_error(error, "media") from None
