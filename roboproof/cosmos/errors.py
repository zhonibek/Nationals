"""Public errors deliberately exclude exception text, credentials and paths."""


MESSAGES = {
    "configuration": "Invalid Cosmos configuration. Check the documented limits and options.",
    "dependency": "Cosmos dependencies are missing or incompatible. Use the scoped requirements with a matching CUDA PyTorch/torchvision pair.",
    "device": "An NVIDIA CUDA GPU and CUDA-enabled PyTorch are required. CPU/ROCm fallback is disabled; no model was loaded.",
    "precision": "This GPU does not support the requested BF16 precision. Select float16 explicitly or use a BF16-capable GPU.",
    "memory": "Insufficient GPU memory for the configured admission threshold. Free GPU memory or use suitable hardware; no CPU offload is allowed.",
    "busy": "Cosmos is already loading or processing one request. Try again after it finishes.",
    "path": "Choose a regular local media file inside an approved upload/media directory. URLs, network paths, links and directory traversal are not accepted.",
    "upload": "The media file is empty, changed during reading, or exceeds the upload size limit.",
    "media_type": "Supported media: JPEG/PNG images and MP4 video files only.",
    "media": "Cannot safely decode this media. Use a valid, non-animated image or a seekable MP4 with a known duration and frame rate.",
    "pixels": "Media dimensions or decoded pixel work exceed the configured limit. Resize the media before uploading.",
    "duration": "Video duration is missing, inconsistent, or exceeds the configured limit. Trim or re-encode it before uploading.",
    "decode_limit": "Video decoding exceeded its bounded frame/time budget. Use a shorter, lower-resolution MP4 with frequent keyframes.",
    "instruction": "Enter a non-empty text instruction within the character limit; control characters and model special tokens are not accepted.",
    "tokens": "Requested output tokens must be an integer within the configured range.",
    "context": "Media and instruction exceed the input token limit. Reduce frames, resolution or instruction length.",
    "cache": "Model files are not available in the local cache. Only the operator may explicitly enable model downloads; imports and UI startup do not download weights.",
    "access": "Hugging Face model access failed. The operator must accept NVIDIA's model access terms and provide an authorized HF_TOKEN environment variable; do not paste credentials into the UI.",
    "load": "Cosmos model loading failed. Check the pinned dependency stack, model access/cache and CUDA hardware. Raw diagnostics are withheld to protect secrets and local paths.",
    "inference": "Cosmos inference failed; no model commentary was produced. Check input compatibility and GPU resources. Raw diagnostics are withheld.",
    "empty_output": "The model returned no decodable commentary. No substitute output or task action was generated.",
    "singleton": "This process already has a Cosmos service with different settings. Use its existing configuration.",
}


class CosmosError(RuntimeError):
    def __init__(self, code: str):
        self.code = code if code in MESSAGES else "inference"
        super().__init__(MESSAGES[self.code])


def sanitized_error(error: Exception, phase: str = "inference") -> CosmosError:
    if isinstance(error, CosmosError):
        return CosmosError(error.code)
    current = error
    for _ in range(6):
        classes = {error_class.__name__ for error_class in type(current).__mro__}
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if "OutOfMemoryError" in classes or isinstance(current, MemoryError):
            return CosmosError("memory")
        if "GatedRepoError" in classes or status in (401, 403):
            return CosmosError("access")
        if classes & {"LocalEntryNotFoundError", "OfflineModeIsEnabled"}:
            return CosmosError("cache")
        if isinstance(current, (ImportError, ModuleNotFoundError)):
            return CosmosError("dependency")
        current = current.__cause__ or current.__context__
        if current is None:
            break
    return CosmosError(phase if phase in MESSAGES else "inference")
