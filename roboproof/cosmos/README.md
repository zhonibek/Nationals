# Real Cosmos commentary runtime

Built on NVIDIA Cosmos. This package loads the actual
`nvidia/Cosmos-Reason2-2B` checkpoint using Transformers
`Qwen3VLForConditionalGeneration` and `AutoProcessor`. There are no mock responses,
fallback task actions, remote inference providers, controller commands, or training
paths in the runtime. Model text is advisory, unverified commentary, not original
Simulator measurements. Robot execution and approval remain outside this package.
The original Simulator/iraLIB physics, controllers, rewards, actions, and gates
are not imported or modified.

Both entry points default to official repository revision
`9ce19a195e423419c349abfc86fd07178b230561`, verified by the parent through
`HfApi.model_info(token=False)` (`model_type=qwen3_vl`, `gated=auto`). Model and
processor receive the same revision. There is no `main`/unpinned default.

## Evidence status

Implementation and dependency-free contract tests do **not** demonstrate live model
inference, CUDA execution, ZeroGPU compatibility, task success, or learned improvement.
The operator must validate actual loading and generation on approved hardware.
No dependency installation, checkpoint download, model call, or server launch is
needed for the contract tests. Importing `space_app` is intentionally different:
it eagerly loads the real checkpoint and must not be used as a smoke import.

## Local / operator-managed NVIDIA host

From the full project root, after the operator has supplied compatible dependencies:

```text
python -m roboproof.cosmos.app
```

This binds to `127.0.0.1:7860`, creates no share tunnel, and loads nothing until
"Load model" or "Analyze with Cosmos" is selected. Only already-cached files are
allowed by default. The operator alone can explicitly authorize a full download:

```text
python -m roboproof.cosmos.app --allow-model-download
```

`--revision` or `COSMOS_MODEL_REVISION` can explicitly select another operator-verified
revision; the default is the verified commit above. An explicit
`--host 0.0.0.0` is available for an approved hosted environment; do not use it on
an unprotected machine. `--media-root DIRECTORY` adds an operator-approved local
read directory; the UI itself accepts file uploads, not path or URL text inputs.
`--upload-root` controls its isolated temporary upload directory. Gradio expires
cached uploads after approximately one hour, with hourly cleanup checks.

The process uses a lazy singleton and rejects overlapping load/generation calls.
The UI has shared concurrency one and a four-request pending queue. NVIDIA CUDA
is mandatory: no automatic CPU/ROCm fallback, CPU offloading or quantization.
Default precision is BF16; unsupported GPUs receive a precision error. Explicit
`--dtype float16` is available but is not claimed to match NVIDIA's BF16 validation.

Admission defaults require 24 GiB total GPU memory and 8 GiB currently free.
These are conservative operator-configurable checks, **not** a guarantee against
OOM. `--min-total-gpu-gib` and `--min-free-gpu-gib` cannot be less than 8 GiB.
The NVIDIA model card currently states a 24 GB minimum; activation needs depend
on media and output length. A roughly 6 GB system-RAM Windows laptop is not an
appropriate full-model execution target. Use the approved NVIDIA host/ZeroGPU.
No workload is launched to probe the local GPU.

`requirements-local.txt` describes a separate Python 3.10–3.12 CUDA environment,
also usable on an operator-managed Lightning host for **inference only**. It
uses Transformers5.19.0 / Torch2.11.0, with matching torchvision0.26.0 and
Gradio6.30.0. PyPI metadata verified that the older Transformers4.57.3 requires
Hub<1 and conflicts with Gradio6.30's Hub>=1.16; the modern stack avoids replacing
the platform-managed Hub. The operator must choose the CUDA wheel source and
driver compatible with the actual host. Gradio 6.x is bounded but not live-tested
here. NVIDIA's model card only lists Linux/BF16 as tested; Windows compatibility
and current patch-level dependency compatibility remain unverified. This package
does not initiate Lightning training or change existing ML implementations.

## Dedicated ZeroGPU entry

`space_app.py` uses the distinct ZeroGPU lifecycle described in the
[official ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu):

- `spaces` imports before Torch or Transformers; missing `spaces` is an error,
  never an invented no-op decorator.
- The real model is loaded eagerly at module scope using BF16 and SDPA, then
  `.to("cuda")` with a string. There is no `device_map` in this adapter.
- Real hardware/memory/precision admission occurs **only** inside the direct
  top-level `@spaces.GPU(duration=120)` function. Module-scope CUDA placement may
  be emulated by the platform and is not presented as GPU execution evidence.
- Gradio registers that decorated function directly, with shared concurrency one.
  Decorator arguments and returned values contain no bound service or lock.
  Returns are strings and JSON-compatible CPU metadata, never CUDA tensors.
- Default ZeroGPU size remains `large` (48 GB according to the current platform
  docs); no `xlarge`, paid inference endpoint, or automatic billing API is used.
  Platform quotas/credit policies still apply; the operator controls the account.

The Space adapter defaults to the verified repository commit
`9ce19a195e423419c349abfc86fd07178b230561`. An override via `COSMOS_MODEL_REVISION`
must also be a full 40-character official repository commit. Model and processor
cannot silently drift to `main`. Both runtimes report the requested and resolved
revision in generation provenance. The earlier weight-only commit is not used
as the default.

Downloads remain disabled unless the operator sets `COSMOS_ALLOW_MODEL_DOWNLOAD=1`
before the controlled Space startup. Without that explicit permission, startup
requires a complete local cache. This intentionally does not download a checkpoint
during unapproved review/import. Access terms for NVIDIA's gated model must be
accepted by the operator. At handoff, the agreement is pending in the user's
browser, no credentials are assumed, and no weights/live inference have been
obtained. `HF_TOKEN` is read only from the process environment,
passed directly to Hugging Face loading, and never logged, written to disk, shown
in status, accepted through the UI, or read from a saved login by this code.

The parent owns Space creation, hardware selection, credentials, metadata, root
entry wrapper, and upload. It should package the full `roboproof/cosmos` directory,
copy `requirements-space.txt` as the Space's requirements, and use a root entry
wrapper that imports `launch` from `roboproof.cosmos.space_app` and explicitly
calls `launch(host="0.0.0.0")`. This avoids assumptions about direct nested-script
package paths. The module is also launchable from the project root with:

```text
python -m roboproof.cosmos.space_app --host 0.0.0.0
```

The Space requirements do not pin or replace platform-managed `gradio`, `spaces`,
`huggingface_hub`, or `torch`. `torchvision==0.26.0` requires the supported
ZeroGPU Torch2.11.0 runtime; its PyPI dependency metadata was verified on2026-10-11.
The actual platform runtime must still be checked before deployment. Transformers5.19.0
is pinned for compatibility with the current platform-managed Hub. Neither Space dependencies nor
native video decoding have been installed/verified here.

## Input and output bounds

Defaults accept one JPEG/PNG image or one MP4 video, up to 32 MiB and 30 seconds.
Images must be non-animated. Video must have exactly one video stream and known
finite duration, time base and frame rate. Sampling aims at 4 FPS, evenly covering
the declared duration, capped at 32 frames. Frame counts are even for Qwen3-VL's
temporal patches; short clips may repeat a real frame, never synthesize one.
The original clip is not silently replaced with only an early prefix.

Native frames are limited to 8,294,400 pixels and aspect ratio 32:1. Model frames
are resized to 32-pixel-aligned dimensions within 262,144 pixels each. Video work
is bounded to 512 decoded frames, 512 million decoded pixels, and a cooperative
15-second budget checked between decoder calls. Native decoder calls themselves
are not forcibly interrupted; this is not an OS sandbox or a hard wall-clock
timeout. Unknown/malformed/expensive clips fail admission rather than being
silently truncated. Audio is not processed. Uploaded media is decoded from a
bounded byte snapshot, using only the MOV/MP4 demuxer, with external data references
disabled and network protocols excluded; no arbitrary URL reaches Transformers.

Instructions are capped at 4,000 characters and reject control/model special-token
injection. The expanded input cap is 8,192 tokens. The UI defaults to4,096 new tokens,
matching NVIDIA's recommendation for its reasoning format. Bounded outputs may
still truncate; the UI/provenance explicitly flags this.
The original decoded model text is returned without a fabricated answer/actions.
Video provenance contains actual sampled timestamps; Qwen3-VL timestamps use
rounded frame indices at the source average FPS, so variable-rate timing is
approximate, not measured robot timing.

Local paths are resolved under approved roots, checked for regular files, size
and identity, and reject URLs, UNC paths, traversal, symlinks and junctions.
An ASGI request guard also rejects remote FileData references before Gradio's
input resource fetching, disables proxy routes, validates file-serving paths,
and caps JSON requests at 64 KiB / one media reference. It does not rewrite model
output. Multipart upload limits remain enforced by Gradio's maximum file size.
Paths, exception details and credentials are excluded from public errors. Existing
local filesystem permissions still matter; this is not protection from an attacker
with permission to replace directories concurrently. Public hosting additionally
requires parent-managed authentication, quotas and retention policy.

## Lightweight contract tests

```text
python -B -m unittest discover -s tests -p roboproof_cosmos_contract_test.py
```

These stdlib-only tests use synthetic decoder/model/service objects solely to
validate admission, limits, sanitized errors, isolation, lazy loading, concurrency
and the ZeroGPU source structure. They are **not real inference tests** and do not
download weights or import the inference stack. Parent-managed live validation
must exercise a small permitted real image and video, actual revision resolution,
OOM/access failures, timestamps, UI uploads, and the ZeroGPU decorated handler.

Primary references: [NVIDIA model card](https://huggingface.co/nvidia/Cosmos-Reason2-2B),
[official inference sample](https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/scripts/inference_sample.py),
[Qwen3-VL processor source](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3_vl/processing_qwen3_vl.py).
