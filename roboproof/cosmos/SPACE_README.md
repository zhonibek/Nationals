---
title: RoboProof Cosmos
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
short_description: Understand robot images and videos with NVIDIA Cosmos
python_version: "3.12"
startup_duration_timeout: 30m
---

# RoboProof Cosmos

Built on NVIDIA Cosmos. This application runs the actual
[`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
vision-language model at revision `9ce19a195e423419c349abfc86fd07178b230561`.
Upload a JPEG/PNG image or MP4 video and ask about visible motion, objects or
possible next steps. Answers come from the model, with no canned responses.

Movement execution and measured robot outcomes belong to the original
[RoboProof Simulator/iraLIB project](https://github.com/zhonibek/Nationals).
This Space returns model observations and plans; it does not command a robot.

## Runtime setup

Select ZeroGPU (`zero-a10g`) in the Space's actual hardware settings; README
frontmatter does not select hardware. Use the supported Torch2.11.0 runtime
with the requirements supplied here. The application imports `spaces` before
Torch and preloads the pinned model at startup; the decorated analysis handler
requests a GPU only for real generation. Hardware and inference must be
validated through successful image/video requests, not the RUNNING label alone.

ZeroGPU's default large slice is currently48GB. Actual hosting eligibility is
account-dependent: this project's account currently shows compute-backed Gradio
creation as paid-plan-only. Do not purchase an upgrade or silently select paid
GPU hardware. No successful deployment or generated answer is claimed by this
package. GPU names and the SHA-256 of the exact decoded input-byte snapshot are
included in generated response provenance; these are not signed hardware
attestations or proof that the commentary is correct.

The operator must obtain NVIDIA model access and supply a model-read credential
as the private `HF_TOKEN` Space secret. Set `COSMOS_ALLOW_MODEL_DOWNLOAD=1`
to authorize downloading that model at startup. Credentials never belong in
repository files or public Space variables. Set `GRADIO_SSR_MODE=false` for
the model-loading startup. No paid GPU or storage is selected by this source.

The code's dependency/admission tests do not establish working hosted inference.
The deployed revision, actual GPU, generated answers, timing and output truncation
must be inspected before claiming the service is operational.
