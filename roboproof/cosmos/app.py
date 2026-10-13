"""Launch the local-only interface with python -m roboproof.cosmos.app."""

import argparse
import importlib
import os
from pathlib import Path
import sys
import tempfile

from .config import MODEL_REVISION, RuntimeConfig
from .errors import CosmosError, sanitized_error
from .request_guard import guarded_app_kwargs
from .runtime import get_service


BOUNDARY_NOTICE = (
    "Built on NVIDIA Cosmos. Ask about an image or video of robot motion. "
    "Answers are model observations; robot movement requires approval in RoboProof."
)


def error_result(error: Exception) -> tuple[str, dict, str]:
    public_error = sanitized_error(error)
    return "", {"status": "error", "code": public_error.code, "message": str(public_error)}, str(public_error)


def create_interface(config: RuntimeConfig):
    service = get_service(config)

    def load_model():
        try:
            return service.initialize(), "Model loaded. Live inference has not been verified by loading alone."
        except Exception as error:
            _, details, message = error_result(error)
            return details, message

    def analyze(media_path, instruction, token_count):
        try:
            result = service.infer(media_path, instruction, token_count)
            message = "Real model generation completed; commentary remains unverified."
            if result["possibly_truncated"]:
                message += " Output reached its token cap and may be truncated."
            return result["commentary"], result, message
        except Exception as error:
            return error_result(error)

    return build_interface(config, analyze, service.status(), load_model)


def build_interface(config: RuntimeConfig, analyze, initial_status: dict, load_model=None):
    try:
        gradio = importlib.import_module("gradio")
    except Exception:
        raise CosmosError("dependency") from None
    limits = config.limits
    with gradio.Blocks(title="RoboProof — Cosmos commentary", analytics_enabled=False, delete_cache=(3600, 3600)) as interface:
        gradio.Markdown("# Cosmos-Reason2-2B\n" + BOUNDARY_NOTICE)
        gradio.Markdown(
            f"JPEG/PNG or MP4; at most {limits.max_upload_bytes // (1024 * 1024)} MiB, "
            f"{limits.max_duration_seconds:g} seconds, {limits.max_frames} sampled frames. "
            f"Instructions: {limits.max_instruction_chars} characters. NVIDIA CUDA only; one request at a time. "
            f"GPU admission: {config.min_total_gpu_gib:g} GiB total, {config.min_free_gpu_gib:g} GiB free. "
            + ("No weights are loaded at local UI startup." if load_model is not None else "ZeroGPU: model preloaded at module scope; real GPU admission occurs only inside the GPU handler.")
        )
        if load_model is not None:
            load_button = gradio.Button("Load model — operator-enabled downloads" if config.allow_model_download else "Load model — cached files only")
        media_file = gradio.File(label="Local image or video upload", type="filepath", file_count="single", file_types=[".jpg", ".jpeg", ".png", ".mp4"])
        instruction = gradio.Textbox(label="Instruction", lines=3, placeholder="Describe the visible motion and uncertainties.")
        if limits.max_output_tokens == 1:
            token_count = gradio.Number(value=1, minimum=1, maximum=1, precision=0, interactive=False, label="Maximum output tokens")
        else:
            token_count = gradio.Slider(minimum=1, maximum=limits.max_output_tokens, value=limits.max_output_tokens, step=1, precision=0, label="Maximum output tokens")
        analyze_button = gradio.Button("Analyze with Cosmos", variant="primary")
        commentary = gradio.Textbox(label="Model commentary — not Simulator evidence", lines=10, interactive=False)
        with gradio.Accordion("Model and runtime details", open=False):
            details = gradio.JSON(label="Runtime provenance", value=initial_status)
        status = gradio.Textbox(label="Status", value="No inference performed.", interactive=False)
        if load_model is not None:
            load_button.click(load_model, inputs=[], outputs=[details, status], concurrency_limit=1, concurrency_id="cosmos", api_name=False)
        analyze_button.click(analyze, inputs=[media_file, instruction, token_count], outputs=[commentary, details, status], concurrency_limit=1, concurrency_id="cosmos", api_name="analyze")
    return interface.queue(default_concurrency_limit=1, max_size=4)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Cosmos commentary; CUDA only, no automatic model download.")
    parser.add_argument("--host", default="127.0.0.1", help="Explicitly set 0.0.0.0 only on an approved host; no public share tunnel is created.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--upload-root", type=Path, default=Path(tempfile.gettempdir()) / "roboproof-cosmos-uploads")
    parser.add_argument("--media-root", type=Path, action="append", default=[], help="Additional operator-approved local media directory.")
    parser.add_argument("--revision", default=os.environ.get("COSMOS_MODEL_REVISION", MODEL_REVISION))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--allow-model-download", action="store_true", help="Explicitly authorize downloading the official model when first loaded, subject to model access.")
    parser.add_argument("--min-total-gpu-gib", type=float, default=24)
    parser.add_argument("--min-free-gpu-gib", type=float, default=8)
    return parser


def main(argv=None) -> int:
    args = argument_parser().parse_args(argv)
    try:
        if args.host not in ("127.0.0.1", "0.0.0.0", "::1") or not 1 <= args.port <= 65535:
            raise CosmosError("configuration")
        args.upload_root.mkdir(parents=True, exist_ok=True)
        upload_root = args.upload_root.resolve(strict=True)
        os.environ["GRADIO_TEMP_DIR"] = str(upload_root)
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        os.environ.pop("GRADIO_ALLOWED_PATHS", None)
        config = RuntimeConfig(
            media_roots=(upload_root, *args.media_root), revision=args.revision,
            device=args.device, dtype=args.dtype, allow_model_download=args.allow_model_download,
            min_total_gpu_gib=args.min_total_gpu_gib, min_free_gpu_gib=args.min_free_gpu_gib,
        )
        interface = create_interface(config)
        interface.launch(
            server_name=args.host, server_port=args.port, share=False, inbrowser=False,
            max_file_size=config.limits.max_upload_bytes, show_error=False, strict_cors=True,
            allowed_paths=[], enable_monitoring=False, ssr_mode=False,
            app_kwargs=guarded_app_kwargs(config),
        )
        return 0
    except Exception as error:
        public_error = sanitized_error(error, "configuration")
        print(f"Cosmos [{public_error.code}]: {public_error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
