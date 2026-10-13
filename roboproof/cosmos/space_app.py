"""Operator-controlled ZeroGPU entry point; importing this module loads weights."""

import os
from pathlib import Path
import re
import tempfile

from roboproof.cosmos.config import MODEL_ID, MODEL_REVISION, RuntimeConfig
from roboproof.cosmos.errors import CosmosError, sanitized_error


UPLOAD_ROOT = Path(tempfile.gettempdir()) / "roboproof-cosmos-zerogpu-uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(UPLOAD_ROOT.resolve())
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ.pop("GRADIO_ALLOWED_PATHS", None)
REVISION = os.environ.get("COSMOS_MODEL_REVISION", MODEL_REVISION)
if not re.fullmatch(r"[a-f0-9]{40}", REVISION):
    raise CosmosError("configuration")
CONFIG = RuntimeConfig(
    media_roots=(UPLOAD_ROOT,), revision=REVISION, device="cuda",
    allow_model_download=os.environ.get("COSMOS_ALLOW_MODEL_DOWNLOAD") == "1",
)

try:
    import spaces
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
except Exception:
    raise CosmosError("dependency") from None

from roboproof.cosmos.app import build_interface, error_result
from roboproof.cosmos.media import prepare_media, validate_instruction, validate_output_tokens
from roboproof.cosmos.request_guard import guarded_app_kwargs
from roboproof.cosmos.runtime import check_gpu, generate_commentary


try:
    PROCESSOR = AutoProcessor.from_pretrained(
        MODEL_ID, revision=REVISION, token=os.environ.get("HF_TOKEN") or False,
        local_files_only=not CONFIG.allow_model_download, trust_remote_code=False,
    )
    MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=REVISION, token=os.environ.get("HF_TOKEN") or False,
        local_files_only=not CONFIG.allow_model_download,
        trust_remote_code=False, use_safetensors=True,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    MODEL.eval()
except Exception as error:
    public_error = sanitized_error(error, "load")
    if public_error.code == "load" and isinstance(error, OSError) and not CONFIG.allow_model_download:
        public_error = CosmosError("cache")
    raise public_error from None


@spaces.GPU(duration=120)
def analyze_media(media_path: str, instruction: str, token_count: int) -> tuple[str, dict, str]:
    try:
        instruction = validate_instruction(instruction, CONFIG.limits)
        token_count = validate_output_tokens(token_count, CONFIG.limits)
        check_gpu(torch, CONFIG)
        prepared = prepare_media(media_path, CONFIG)
        result = generate_commentary(torch, MODEL, PROCESSOR, prepared, instruction, token_count, CONFIG)
        message = "Real ZeroGPU model generation completed; commentary remains unverified."
        if result["possibly_truncated"]:
            message += " Output reached its token cap and may be truncated."
        return result["commentary"], result, message
    except Exception as error:
        return error_result(error)


interface = build_interface(
    CONFIG, analyze_media,
    {"status": "preloaded_not_inference_verified", "model_id": MODEL_ID, "requested_revision": REVISION, "hardware_admission": "inside_GPU_handler_only", "robot_actions_executed": False, "training_performed": False},
)


def launch(host: str = "127.0.0.1", port: int = 7860):
    if host not in ("127.0.0.1", "0.0.0.0", "::1") or type(port) is not int or not 1 <= port <= 65535:
        raise CosmosError("configuration")
    return interface.launch(
        server_name=host, server_port=port, share=False, inbrowser=False,
        max_file_size=CONFIG.limits.max_upload_bytes, show_error=False,
        strict_cors=True, allowed_paths=[], enable_monitoring=False, ssr_mode=False,
        app_kwargs=guarded_app_kwargs(CONFIG),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Operator-controlled Cosmos ZeroGPU entry point.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    arguments = parser.parse_args()
    launch(arguments.host, arguments.port)
