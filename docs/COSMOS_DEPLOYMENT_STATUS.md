# Cosmos deployment and T4-only Nemotron

## Current state — 2026-10-16

- The existing `roboproof.cosmos.space_app` runs the pinned NVIDIA Cosmos-Reason2-2B adapter, not a substitute model. Successful GPU inference has **not** been obtained in this continuation.
- The signed-in Hugging Face account's New Space page disables Gradio/Docker creation and requests PRO. No subscription, paid GPU, public repository or access-control bypass has been attempted. Public documentation describes eligible free accounts, but that does not override this account's visible restriction.
- At09:11:45 UTC, a newly executed hardware-only cell in the approved private Colab notebook reports an actual Tesla T4 with15,637,086,208 bytes, Torch2.11.0+cu130/CUDA13.0, Python3.13.15, NumPy2.1.3, two CPUs and Node20.19.0. The receipt is retained in the ignored cache. This is fresh allocation metadata, not a historical notebook output or proof of model generation/training. The earlier selector enabled T4 but disabled L4/A100/H100; no paid upgrade was selected.
- NVIDIA's model card specifies at least24GB GPU memory. The normal Cosmos BF16/24GiB admission remains unchanged. The user explicitly approved a separately labelled small-input FP16/T4 experiment without CPU offload. That experiment has not loaded the model or generated commentary yet; it must not be presented as the documented24GB configuration, successful deployment or training.
- Nemotron must use an explicitly checked NVIDIA T4. The old Windows CPU launcher now fails closed before reading weights or starting a process. The loopback client may connect through an authenticated operator-controlled tunnel, but neither an endpoint alias nor a connection check proves GPU identity. No T4 Nemotron service is connected or claimed here.

## Dashboard integration

The Cosmos panel reads saved answers through the local read-only
`/api/cosmos/results` endpoint. Loading or refreshing it never starts inference,
training, an upload or a paid service. The result reader admits bounded records
with the pinned checkpoint, input hash, CUDA device metadata, source motion
session and explicit advisory-only flags. It rejects loaded-only statuses,
missing answers, CPU results, mismatched inputs/models and robot-execution or
training claims. Record validation is not independent hardware attestation or
semantic verification of a model's answer.

Actual generated responses now include the CUDA-reported GPU name and SHA-256
of the same immutable input-byte snapshot used for decoding. No fabricated
Cosmos text or successful example is prepopulated in the UI. No movement,
controller, PPO/TD3 objective or qualification gate changes for this feature.

## Prepared original movement inputs

Two already-viewed original Simulator/iraLIB sessions are packaged as32-frame
schematic pose visualizations in the ignored `.cache/cosmos-motion-media`
directory: the first declared simple development example and the previously
selected longer example, including its actual deadline failure. These are not
camera recordings or newly simulated motion. Source session bytes, every PNG,
zero-based transition/post-step timing and encoding time basis are preserved.
No inference or fresh Simulator query occurs during export.

Use `python -m roboproof.cosmos.motion_media --session <saved-session-uuid>` to
prepare another explicitly selected saved session. Existing exports cannot be
overwritten. Use `python -m roboproof.cosmos.package_space` to build an
allowlisted deployment ZIP, without logging in, copying tokens, publishing or
allocating compute. Package only the Cosmos adapter; never upload this entire
worktree, saved credentials or personal files into a public Space.

## Required next evidence

1. An actually available, approved no-extra-charge GPU allocation, with exact hardware checked.
2. Model access on that host, without posting credentials in chat or Git.
3. Real generation on the prepared input, with exact checkpoint/input/GPU provenance and retained failures/truncation.
4. A complete saved result imported into the original dashboard; no canned replacement if generation fails.
5. Separate independent movement-learning qualification. VLM commentary is not proof of better motion, game scoring or physical-robot safety.

References: [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu),
[NVIDIA model card](https://huggingface.co/nvidia/Cosmos-Reason2-2B).
