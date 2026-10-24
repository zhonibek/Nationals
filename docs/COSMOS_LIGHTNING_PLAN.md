# Cosmos and Lightning AI

## Verified setup — 2026-10-11

The authorized Lightning organization showed 5.00 credits. Its existing `scratch-studio-devbox` was started on the machine explicitly marked **Free, 4 CPU**, with the existing ten-minute inactivity sleep. The published hackathon working branch was cloned into `/teamspace/studios/this_studio/Nationals-work3` at `c88626d`. The original `motion-check` reported `readyForBoundedCpuExperiment: true`. PyTorch imports; its CUDA build is installed, but `cudaAvailable` is false on this CPU machine. No GPU workload, Cosmos inference or AMD result is established by that check.

The machine chooser advertised single-device L4 24 GB at 1.68 credits/hour, A100 40 GB at 2.19, L40S 48 GB at 3.54, H100 80 GB at 5.68 and H200 141 GB at 4.50. These are observations in this account, not guaranteed prices or availability. Only the CPU was requested. Both the saved Hugging Face login and configured token were absent in the inspected environment; no credential was read or created.

A bounded original-engine TD3 CLI readiness run at seed109001 completed four updates on that free CPU:222 accepted rows/292 actual decisions, zero discarded work. Its manifest reports CPU execution, PyTorch `2.14.0+cu130` and `learnedImprovementVerified: false`. The checkpoint, actor and manifest remain in `roboproof/runs/motion/research/lightning-cpu-readiness-109001/` in the Studio. This is actual neural training through the original engine, not a GPU test or independent movement evaluation.

After that run, the dashboard still showed5.00 total credits and the Studio onFree4 x CPU. The actual browser's official Cosmos checkpoint page is logged out and shows the model-access/contact-sharing requirement. Prepare that access before requesting a GPU. Include the new cloud exposure journal in the exclusion inventory before a future independent final.

Git does not transfer ignored local `roboproof/runs/` history or downloaded model caches. Before resuming historical studies or preparing a final on this new host, migrate and hash-check the relevant checkpoints, exposure journals, frozen/consumed corpora, protocols and negative results separately. Exact resumption also requires the recorded runtime/source/configuration; a different PyTorch build must not silently inherit an old study's reproducibility claim.

The cloud readiness archive was subsequently verified and imported back into the local research inventory, preserving all11 files and the original checkpoint. Its four actual world seeds/geometries and changed-seed/deadline aliases are rejected by the existing final exclusion audit. The transfer in the opposite direction, including historical local consumed-final/training evidence, is still required before preparing an independent final in Lightning. A screenshot or successful source clone cannot replace that evidence migration.

## Intended role in the existing project

Cosmos can support visual analysis of robot motion, annotations for recorded demonstrations, and proposed higher-level plans. The existing motion learner still learns four bounded reference actions; the original Simulator/iraLIB executes them and determines physical outcomes. A generated video or proposed action never supplies a verified success, collision or reward label. Keep the original simulator as the evaluator and retain unsuccessful trials.

The current raw34 sensor/reference interface has no camera input. First prepare synchronized original-Simulator RGB frames, timestamps, public sensor observations and actual actions from separate training episodes. Bind each recording to the original engine/controller/source identities. Ground visual annotations against recorded simulator events; do not expose evaluator-only pose or holdout cases to a deployed model. Existing consumed finals stay excluded from training and model prompts.

A first bounded Cosmos experiment should analyze recorded motion and return commentary without motor, shell or automatic movement tools. Movement-policy improvement remains the prerequisite for strengthening tactical execution. UX, voice and avatar changes remain plan-only.

## Model choices

- **Cosmos-Reason2-2B:** a smaller initial candidate for recorded image/video analysis. NVIDIA documents a 24 GB GPU minimum; validate the actual device, free VRAM, runtime and representative clip before claiming support. Its official checkpoint currently requires Hugging Face access approval and sharing contact information with NVIDIA.
- **Cosmos 3 Nano Reasoner:** a newer reasoning option. The official NIM matrix specifies 23.1 GiB per device, architecture/precision constraints and practical host RAM of 24 GiB for BF16 or 18 GiB for FP8/NVFP4. The runtime also reserves GPU memory. A nominal 24 GB device alone is insufficient evidence of a fit.
- **Cosmos 3 Generator / action adaptation:** a later research stage. Nano generation requires at least 31 GiB GPU plus 64 GiB host RAM in the documented layer-offload configuration; resident configurations need more VRAM. Action adaptation needs embodiment-specific action-labeled data and evaluation against this robot's bounded interface. Generic pretrained actions cannot be treated as a trained VEX controller.

## Resource sequence

1. Prepare source, dependencies, model access and recordings on the free CPU Studio. Check compatibility and available storage before downloading multi-GB checkpoints.
2. Obtain the user's model access/license approval through Hugging Face. Keep credentials in the provider's private environment; never in Git, reports, screenshots or chat.
3. After all CPU preparation is complete, select the documented compatible GPU within the approved budget. The user's latest preference is a single B200, not T4. Confirm current credits, rate, free VRAM, RAM and auto-sleep. Use only existing free credits; no purchase, paid plan or auto-top-up.
4. Predeclare one short inference experiment, a hard process time limit and an upper credit allowance derived from the displayed rate. Save model revision, actual device/runtime, input hashes, outputs, timing, truncation and failures.
5. Stop the GPU or switch back to the free CPU immediately after the experiment, and verify that state. Recheck credit use; a process exit alone does not stop machine billing.
6. Compare annotation/plan quality against original recorded events. Only after measured usefulness consider a reviewed adapter to the existing task pipeline. Then retain the full motion and tactical gates.

PPO still uses one CPU thread. TD3 now has an explicit single-NVIDIA CUDA neural path, while original Node/WASM Simulator/iraLIB collection remains on CPU. A GPU allocation alone does not establish successful tensor operations or improvement. Checkpoints bind execution runtime and CUDA RNG; actual operation, continuation and trajectory/evaluation comparisons are still required. The existing HIP/ROCm preflight intentionally rejects a CUDA build; CUDA availability must not be relabeled AMD readiness.

## Subsequent implementation and access checks

The user completed local Hugging Face device sign-in and personally submitted NVIDIA's model-access agreement. An authenticated metadata request then verified access to the pinned Cosmos revision. The private ZeroGPU Space creation request was rejected with HTTP402: this account must satisfy the provider's PRO/account-age requirements. No Space was created, paid plan purchased or alternative paid hardware requested for that deployment.

The existing project now contains the real Cosmos Transformers loader, bounded JPEG/PNG/MP4 preprocessing, local Gradio interface, stable `/analyze` API and distinct eager ZeroGPU entry point. Transformers5.19.0 is selected to avoid the older Hub<1 dependency conflict with Gradio6.30. The32 passing dependency-free contracts plus one host symlink skip do not prove inference. Two additional actual Pillow/PyAV decoder tests and two live local Gradio upload/HTTP-guard tests pass without loading weights. Local startup and `/analyze` were verified with Gradio6.30.0. No canned response or fallback model is supplied.

Installed compatible inference dependencies in Lightning's supported default environment on its free CPU, preserving the prior package inventory. Creating a second virtual environment was refused by Lightning; that restriction was not bypassed. Copied28 source/test files only after checking original base hashes and the archive SHA `1df36c36a9964f01cacfabe5e479a5416cf600446fdc371af60c247016edf1ea`, preserving backups and a deployment receipt. The updated original `motion-check` reports readiness. Credentials and historical local training/final evidence were not transferred; the receipt explicitly prohibits independent final preparation there.

The separate Hugging Face device logins inside Lightning expired without approval. No token was copied from the local machine. Cosmos weights/live inference still require that cloud sign-in. Do not repeatedly reserve GPUs while waiting for it. Movement CUDA readiness can be checked separately without any model-download credential: one predeclared T4/four-CPU request at0.55 credits/hour, at most900 seconds/~0.1375 credits from the existing free balance, bounded four-update width128 training, no final evaluation. Actual allocation, operations, saved evidence and shutdown must be inspected; a request is not execution evidence. The T4 is below the documented24GB Cosmos minimum and must not be claimed as its supported host.

## Private free Colab execution — 2026-10-12

The user authorizes private Google Colab notebooks and upload of only Nationals-work3 training code, original Simulator/iraLIB and simulated data. No existing API keys, personal files, public sharing, paid plan or Drive-wide mount is used. The notebook has restricted access and one owner. H100/A100/L4 choices are disabled; the actually usable free option isTeslaT4. Lightning remains stopped after its payment-verification restriction; it does not supply CPU workers to Colab.

A fresh Colab runtime verifies actual CUDA forward/backward and original-engine readiness. The543-file hash-bound source preserves the compiled persistent C++iraLIB controller. A bounded256-width project full4 versus prefix2/resume4 run then verifies every decoded model/critic/target/optimizer/replay/anchor/RNG/history/work state and the actor SHA `ffab5b901aefc72d99cefd3490bd80d819e14fd4bce053af2150433e005fe864`. It collects169 rows/264 real original-simulator decisions/four worlds with zero discarded work, on genuineTeslaT4/Torch2.11.0+cu130/CUDA13. Neural operations run on GPU; original Node/WASM physics and controller execution stay on Colab CPU. This is reproducibility evidence, not learned-motion improvement, Cosmos generation or physical proof.

The complete31-file checkpoint/actor/exposure export is21,000,706 bytes, SHA `911ca36d2bd32a04d299fa5b289bb8e7d54b61a08ae6c28016fc0ac19b643268`. The archive is now saved locally with exact total and per-file hashes; both decoded checkpoints and actor exports match. Runtime/device metadata remain unchanged. All actual training/readiness queries enter the local seed/geometry exclusion ledger. The user approves deleting the temporary runtime after preservation; the notebook remains saved and the GPU is disconnected. The CPU/Torch2.8-qualified evaluator must not accept CUDA records by rewriting their device/runtime fields. GPU evaluation requires explicit same-runtime admission and controls with the original unchanged gates. FreeT4 also does not satisfy the current strict24GB Cosmos admission; no vision-model generation is claimed.

## References

- [NVIDIA Cosmos 3 architecture and action adaptation](https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/)
- [Official Cosmos 3 NIM hardware matrix](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/nim/support-matrix.md)
- [Official Cosmos-Reason2-2B checkpoint and access conditions](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
- [Lightning billing and free-credit rules](https://api.lightning.ai/docs/platform/overview/faq/billing)

These resource references were inspected on 2026-10-11; account observations are dated individually. No trained-motion improvement is claimed.

## Direct-access and B200 preparation — 2026-10-12

The earlier T4 movement check actually completed: one genuine TeslaT4, CUDA13/Torch2.14, four128-width updates,158 accepted rows/237 original-engine decisions, positive69,672,448-byte peak CUDA allocation. The conditional native CUDA test ran and passed. The GPU was stopped; the current account balance is4.870138888888893 credits. Imported metadata/exposure hashes bind the four actual training worlds and exclude their seed/deadline aliases. The GPU actors/checkpoint remain in private cloud storage; a metadata import is not full checkpoint resumption or improvement proof.

Cloud Hugging Face sign-in now succeeds. All15 pinned Cosmos snapshot files, including actual safetensors, downloaded to the private ignored cache. Real cached Qwen3VL processor handling succeeds on a controlled PNG and two-second MP4. These are explicitly synthetic controls, not original-Simulator recordings. Image/video generation through the actual GPU and project `/analyze` API is still pending until its receipt is captured.

The web editor repeatedly stalled/crashed. Do not allocate expensive hardware when shutdown cannot be controlled reliably. The user authorized direct access and local login storage. Installed the officialLightningSDK2026.10.1 in an isolated ignored local environment; completed its official sign-in without printing credentials. The supported direct command/file API works and verifies the Studio's free4CPU state. An initial Windows path-normalization upload mistake was repaired only for the two owned files; no third-party SDK was modified. The new14-file overlay is hash-verified, backed up and original-engine ready, with exact training-source binding.

The user explicitly approved transferring the existing private Studio, project files and saved HF login to Nebius for1.75 credits. The live catalog confirms one B200/20CPU at9.86 credits/hour, with auto-top-up disabled. The combined upper allowance is4.215 credits including transfer and at most900 running GPU seconds. The transfer actually completes, consuming1.74220085725 credits;3.127938031638893 remain. The initial switch stalls amid network failures and its guard verifies the Studio stopped on Nebius. A direct B200 retry and canonical SDK-enum retry consume no additional credits; the server returnsHTTP403 requiring a verified payment method for free-tier compute. No B200 tensor execution, Cosmos answer or four-way cloud training is established.

The user explicitly declines adding a payment method and asks to keep preparing locally. Do not retry compute, add a card, purchase credits, enable top-up or switch providers to evade that requirement. The Studio is verified stopped with no pending allocation. Preserve all failed receipts, cached model and source in the private Studio. Continue a separately predeclared sequential CPU alternative against the already reproduced current-source control; only256/late-teacher/update64 may advance. No new GitHub publication, public endpoint, final preparation or policy promotion occurs.
