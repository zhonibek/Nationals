# Local Nemotron inside RobotAI

Nemotron provides ordinary local conversation and a separate high-level task coordinator, not the motion learner. Its five reviewed [Agent Skills](skills/README.md) inspect robot configuration, prepare bounded reach-pose proposals, diagnose selected saved experiments, review PPO learning evidence and advise on game tactics. Execution still waits for a separate approval and uses the existing `simulator/engine.js` and source-checked iraLIB C++ WASM controller. No independent robot physics, gain changes or firmware access are added.

## Start and try it

From the repository root, on Windows x64 with Node.js 22 or newer:

```powershell
./tools/setup-nemotron.ps1
./tools/start-nemotron.ps1
node roboproof/server.js
```

Setup downloads approximately 2.84 GB of model weights plus the portable CPU runtime. It pins official artifact revisions and checks SHA-256 before use; downloads can resume. Files stay under ignored `.cache/robotai-nemotron/`. It does not install a system service, change PATH, buy credits or register a cloud account. Review the [NVIDIA model license and model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF) before redistribution.

Open [RoboProof / RobotAI](http://127.0.0.1:8766/#nemotron):

1. **Check connection** verifies that the configured model alias is served. Availability alone is not inference evidence.
2. Type an ordinary question and click **Send**, or press Enter. Shift+Enter inserts a newline. Read the reply in the conversation and send a follow-up. **Stop** cancels a pending request; it does not fabricate a reply. Answers appear after completion, not token by token; CPU inference can take a minute or more.
3. The latest chat restores on page load without inference. **Restore chat** reads it again; **New chat** starts a separate conversation without deleting old files. Completed turns save atomically in ignored `roboproof/runs/nemotron-chat/<uuid>.json`. The privacy details include **Export chat**. This history is not training or verified movement evidence.

To prepare movement rather than only discuss it:

1. Expand **Robot tools**. Enter `From (0,0), move to x=24 inches, y=0 inches, heading 0 degrees. Use a 10-second deadline.` and click **Prepare simulation from my message**. If the input is empty, this explicitly selected action uses the latest sent chat message, not a model reply or inferred coordinate.
2. Read the plain-language proposal, coordinates and deadline. Exact JSON, evidence and tool logs are under expandable details. Schema validation does not prove that a model understood your intention correctly.
3. Click **Approve & run in original Simulator**. A bounded worker executes the actual shared engine and independently checks exact same-runtime replay. The displayed outcome and metrics come from that run, not from the model's narration.
4. **Open original 2D / 3D Simulator**, then **Load Nemotron task** and the normal Play button, to visualize the same task through the existing project UI. This is a separate visual execution, not an exact replay of the server report. Its native routine completion is not the benchmark's settling/deadline gate. The 3D view uses the existing public Three.js CDN assets.

**Restore saved experiment** restores the latest saved model proposal/result after browser or server restart. **Download session & evidence** in the tool details exports its JSON. Experiments persist separately as `roboproof/runs/nemotron/<uuid>.json`; approvals, tool evidence, provider identity, token usage, inference wall time and measured report/replay are saved. Tool replies are separate from ordinary chat memory. Model weights are pretrained and unchanged; saved sessions are not training checkpoints. Completed run approvals are idempotent.

## Conversation versus robot tools

Ordinary chat uses one authenticated local completion without function calls, forced skill selection or simulator execution. Only the server-owned saved user/assistant messages enter conversational history; API clients cannot supply assistant/system messages, paths, provider endpoints or tasks. Answers may be wrong; without an explicitly selected robot-tool workflow, the model cannot read current robot settings, saved training results or telemetry.

The small 4,096-token runtime uses a conservative recent-context budget: up to three previous complete turns plus the current message, within 2,400 UTF-8 bytes, and at most 512 output tokens. A message is at most 2,000 characters and 2,400 UTF-8 bytes. Older messages remain visible/saved but may fall outside model context. A saved chat holds up to 50 completed turns; start a new chat after that. Output-limit truncation is visibly labeled. No chat content is parsed into a movement command or approval, and chat UUIDs cannot be approved as experiments.

The interface is conversation-first, not a claim of ChatGPT/Claude model quality. The same local Nemotron weights and CPU runtime remain in use; this change adds neither GPU support nor training.

## Tactical adviser, not autonomous play

Under Robot tools, **Explain game rules & tactics** invokes the fifth skill. In the original Simulator served through RoboProof, start Override and click **Спросить тактический AI** to explicitly supply a compact current snapshot. The adviser reads `get_game_rules` and `get_game_snapshot` before finishing through read-only analysis. It uses a reviewed Override v2.0 subset and original engine constants/source hashes, not unrestricted manual ingestion or a learned winning strategy. The full normative PDF could not be fetched during this update.

Snapshots are bounded, fresh at submission and structurally checked, but browser-supplied—not independently authenticated/replayed facts. Nearby objects are partial visibility; state may change during slow inference. The model cannot access another board, dispatch motion/manipulation/perception, run a game, predict verified score or approve a snapshot as movement. The simulator displays advice as plain text and has Stop AI; replay/standalone-file views cannot call this workflow. [Hierarchy and remaining gates](../docs/TACTICAL_AI_PLAN.md).

The portable model server runs in the background. Closing the browser does not stop it. The start script prints its PID; stop that specific process when finished. Do not start another copy on the same port. Stop the foreground RoboProof server with Ctrl+C. Neither server starts automatically after reboot.

## Model and runtime

- NVIDIA Nemotron 3 Nano 4B, official `Q4_K_M` GGUF.
- Model revision, filename, size and SHA-256; official llama.cpp Windows CPU release and archive SHA-256: `nemotron-runtime.json`.
- CPU inference, 4 threads, one slot, 4,096-token context. No ROCm/CUDA or GPU training is claimed.
- Prompt-cache RAM is explicitly capped at 128 MiB instead of the pinned runtime's 8 GiB default. `start-nemotron.ps1 -CacheMiB 0` disables that cache; allowed settings are 0..1,024 MiB, never unlimited. This is not a total-memory cap: the multi-GB model and other applications can still exhaust this 6 GB host. Avoid running the heavy test/training suite concurrently with local inference.
- Native function calls over a local OpenAI-compatible endpoint. llama.cpp runs with `--jinja`; requests disable thinking through `chat_template_kwargs.enable_thinking=false`. See the [official function-calling documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md).

Optional configuration is server-side, before starting RoboProof:

```powershell
$env:ROBOTAI_NEMOTRON_BASE_URL = 'http://127.0.0.1:8080/v1'
$env:ROBOTAI_NEMOTRON_MODEL = 'robotai-nemotron'
$env:ROBOTAI_NEMOTRON_TIMEOUT_MS = '120000'
node roboproof/server.js
```

The default portable endpoint automatically reads the generated private local key from `.cache/robotai-nemotron/local-api-key.txt`. A separately managed compatible endpoint can use `ROBOTAI_NEMOTRON_API_KEY`; never paste keys into a task prompt, commit them, or put them in browser URLs. Changing the model server port also requires changing the base URL. Other compatible servers must expose the selected Nemotron model ID and native tool calls; those configurations have not been live-tested here.

## Bounds and security

- Model endpoint must be loopback `/v1`; foreign hosts, credentials in URLs, redirects and remote cloud endpoints are rejected. Nebius and paid APIs are disabled, regardless of existing cloud credits.
- The model runtime has a generated private API key, restricted CORS, disabled web UI and disabled built-in agent/command tools. No key is sent to the browser or saved in sessions.
- In the separate robot-tool workflow, the catalog is initially metadata-only. `load_skill` activates at most two reviewed skills; their fixed resource/task tools are `get_robot_profile`, `get_saved_motion_evidence`, `get_learning_summary`, `get_motion_contract` and `prepare_reach_pose`. `ask_clarification` safely ends a request without a task; `finish_analysis` requires real tool evidence. Native function calls are required for that workflow, and preparation is hidden until the contract is read. Unloaded tools and plain-text preparation claims fail closed. Only fixed nominal configuration and user-selected compact saved evidence can be read; no arbitrary shell/files/network, custom handlers, motor-voltage output or physical robot endpoint.
- Prompt: 1..4,000 characters. At most six model turns; 1,024 output tokens per turn; one tool call per turn; no automatic inference retries. Per-request timeout defaults to 120 seconds, configurable from 1..300 seconds. The server's total planning budget is 300 seconds.
- Provider output is capped at 256 KiB. Truncated, parallel, duplicate, malformed or unsupported tool calls fail closed; no fallback rule-based output is labeled Nemotron.
- Task validation is shared with `simulator/motion.js`: X/Y within ±60 inches, heading within ±180 degrees, deadline 0.01..60 seconds in 0.01-second increments. Seed is fixed at 42; no model-selected physics changes.
- Loopback Host/same-origin checks, bounded JSON requests and a single active API operation protect the local dashboard. Cancelling inference aborts the request; simulation requires a separate approval and has its own cancellable worker/time budget.
- Task text, skill catalog/selected instructions and requested bounded tool evidence are supplied to this local model. Diagnosis requires explicitly selected saved-run evidence; at most five telemetry samples enter that snapshot. Full source, credentials, arbitrary files and camera data are not supplied. Local prompts/reports can still contain private information, so review before sharing session exports. New approved runs retain transitions locally; saved sessions use compact JSON with a 16 MiB limit.

## Evidence and remaining work

On 2026-10-01, the verified official model ran on this computer: AMD Ryzen 3 5300U, approximately 6 GB usable RAM, CPU-only. A real two-call inference proposed `(0,0,0°) -> (0,24,0°)` with a 10-second deadline. First observed inference wall time was 153.73 seconds; this is one cold/partly contended measurement, not a throughput guarantee. CPU latency is substantial on this machine.

The approved original-engine run reached the benchmark success gate after 2.91 simulated seconds, with position error 0.0012101607666282113 m, zero recorded contact seconds, and exact replay. This is a scripted-controller result selected through real model inference, **not learned movement improvement**. Native visual execution reached Y=24.0 inches and displayed `Auto Complete / Settled`.

A second real request through the dashboard proposed `(0,0,0°) -> (24,0,0°)` correctly, took 73.7589923 seconds with a warm prompt cache, and also passed the approved original-engine run and exact replay. Session IDs: `4633ddf5-9db8-4f04-8229-68eaac6b9541` and `a66b49c1-9a84-455c-9121-802c7492d838`. Two simple tasks are functional smoke evidence, not a model-quality, motion-efficiency or Qwen comparison benchmark.

The full Node/controller/Simulator suite passed 85 tests with zero failures and one existing external-Java test skipped because a usable Java compiler was unavailable. Generated robot configuration checked successfully. Focused integration/engine tests passed 26/26 without skips. Browser validation found and corrected a WASM-specific CSP loading restriction; the corrected original Simulator loaded the real C++ controller. The local model rejects unauthenticated inference with HTTP 401. Model weights, API key and generated sessions are ignored by Git.

Automated fixtures in `tests/nemotron.test.js` cover transport, validation, budgets, approval, persistence, original-engine execution and replay. Their mocked provider responses are explicitly labeled fixtures; only the retained live sessions establish actual model execution.

Current motion work has since delivered the persistent-controller short-horizon interface and separate CPU PPO learner, but its first frozen improvement gate is negative; see [motion learner](motion_learning/README.md). Agent Skills are workflow specialization, not neural movement improvement. Next: measure skill selection/grounded explanations and develop movement on training-only cases with an untouched final evaluation. Camera grounding/calibration, obstacle planning, VEX game-score rewards, remote Nebius/AMD execution and physical deployment remain separate, unfinished gates.
