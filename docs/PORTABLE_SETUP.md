# Portable Windows setup

Run these PowerShell commands from the repository root unless stated otherwise. They describe explicit setup and validation, not results already obtained on your computer. No robot, cloud account, paid service or GPU is needed for the local simulator and CPU motion learner.

## 1. Prerequisites and a fresh clone

- Windows x64, PowerShell, Git, and a modern browser with WebAssembly support.
- Node.js 24 on `PATH`, matching the major version in `.github/workflows/robotai-tests.yml`. Keep the exact Node version when moving saved runs: their identity also includes V8, platform and architecture.
- Python 3.12 x64 with `venv` and `pip`, accessible as `python`. This is the Python version used by the same workflow. If only the Windows launcher is available, substitute `py -3.12` for base-interpreter `python` commands below; virtual-environment commands remain unchanged.
- Disk space for the clone, a CPU PyTorch environment and saved runs. Optional Nemotron needs another approximately 2.84 GB for its model, plus its runtime archive, extracted runtime and working memory.

From your chosen parent folder:

```powershell
git clone --branch Хакатон https://github.com/zhonibek/Nationals.git Nationals-work3
Set-Location .\Nationals-work3
git status --short
git rev-parse HEAD
node --version
python --version
```

Use the reviewed project revision containing the scripts described here. A remote clone includes committed files only: it does not reproduce another checkout's uncommitted edits, ignored environments, model downloads or training history. For restoration, obtain the matching reviewed source snapshot as well as its artifacts; a commit ID alone cannot identify a dirty checkout. Preserve existing changes rather than resetting or cleaning the repository.

The local Node server and motion bridge use Node's built-in modules; there is no `npm install` step. PROS, an ARM toolchain and Zig are not needed just to start the included simulator. `simulator/control.wasm` and `simulator/control-build.json` are supplied together.

## 2. Start the dashboard and original simulator

Start RoboProof in a foreground terminal:

```powershell
node roboproof/server.js 8766
```

Open [Motion lab](http://127.0.0.1:8766/#motion). The same server serves the [original simulator](http://127.0.0.1:8766/simulator/); a second server is unnecessary for that route. Empty training history on a fresh clone is expected. Opening the dashboard or refreshing training history does not install dependencies, train a policy or call Nemotron. Experiments and comparisons require explicit actions.

For a standalone simulator, use a second terminal at the repository root:

```powershell
.\tools\start-simulator.ps1
```

Open [standalone simulator](http://127.0.0.1:8765). The script prefers an existing app-bundled Python if present, otherwise `python` on `PATH`. To choose your own portable interpreter explicitly, use the equivalent command instead of the script:

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory simulator
```

Do not open `simulator/index.html` through `file://`: WASM loading requires HTTP. The 3D view loads Three.js and loaders from public CDNs; it is not fully offline. The Node/Python headless motion path does not depend on those browser CDN assets. Keep servers bound to loopback. Stop foreground servers with Ctrl+C.

## 3. Create the CPU training environment

This section installs packages and can download a large CPU wheel. Perform it deliberately, once, on the target computer; do not copy a virtual environment from another checkout.

```powershell
python -m venv .\roboproof\gpu\.venv
& '.\roboproof\gpu\.venv\Scripts\python.exe' -m pip install -r .\roboproof\gpu\requirements-cpu.txt
& '.\roboproof\gpu\.venv\Scripts\python.exe' -m pip check
& '.\roboproof\gpu\.venv\Scripts\python.exe' -c "import sys, torch; print(sys.version); print(torch.__version__); print('HIP:', torch.version.hip, 'CUDA:', torch.version.cuda)"
```

If `.venv` already exists, inspect its interpreter and packages before reusing it; do not recreate it over another person's environment. The requirements file pins `torch==2.8.0` through the CPU wheel index. A version such as `2.8.0+cpu` is the CPU build, not a GPU installation. NumPy alone is insufficient. Invoking the environment's executable directly avoids activation-script execution-policy problems.

The canonical learner is `roboproof.motion_learning.train`, not the separate historical failure-prediction learner under `roboproof/ml`. It exchanges observations/actions with the original simulator through a local Node child process and the source-checked C++ WASM controller. Node must remain on `PATH`; an explicitly configured `ROBOTAI_NODE` executable must also point to the intended Node version.

### Explicit readiness and bounded training

Run one resource-intensive operation at a time, especially on a 6 GB computer. Stop local Nemotron before training or running the learning tests. These commands execute simulation/training; they are not merely setup inspection:

```powershell
node tools/generate-config.js --check
node roboproof/cli.js motion-check
& '.\roboproof\gpu\.venv\Scripts\python.exe' -m roboproof.motion_learning.train --seed 42 --seed-count 3 --updates 8 --max-steps 4096 --rollout-steps 128 --max-seconds 180
```

Inspect `motion-check`'s retained evidence and the trainer's checkpoint/finished output. Readiness checks the bounded reach environment and replay; it is not a learned-improvement, full-regression or physical-safety certificate.

The example requests seeds 42, 43 and 44, up to eight updates and 4,096 collected policy steps per seed, with one shared 180-second collection deadline. It uses one bridge and one PyTorch CPU training thread. Complete-episode rollouts may exceed the requested rollout length; unfinished rollouts are discarded. A short budget can leave some seeds with no gradient update. An already-started bounded update, checkpoint or teardown can finish after the collection deadline: `--max-seconds` is not a hard process kill or memory cap.

Saved outputs are under `roboproof/runs/motion/learning/`. Start/return to the dashboard and select **Refresh training history** to read them. Controller-only remains the default. Neither opening the page nor refreshing starts Python training.

### Evaluation is a separate, explicit operation

After training has stopped, and only when ready to inspect the frozen comparison:

```powershell
node roboproof/cli.js motion-compare-learned
```

This executes the controller, initial untrained actors and learned actors on the 24 frozen suite-v2 worlds, then writes the selected run's `evaluation.json`. Archive existing evaluation evidence before repeating it: the same filename is reused. The command selects the run named by `learning/latest.json`, not the best seed or the newest folder timestamp.

The existing gate requires each of three independent training seeds to improve success over both controller and untrained policy, preserve every baseline success, introduce no per-world contact increase, and achieve at least 5% all-world mean time or effort-proxy improvement. Keep failures and negative comparisons. Gradient updates, changed weights or higher training reward do not prove improvement, GPU execution or readiness for a physical robot. Do not weaken thresholds or tune against inspected frozen cases and call the same corpus fresh held-out evidence. See [learner limits](../roboproof/motion_learning/README.md).

## 4. Transfer checkpoints and history

Git ignores `roboproof/runs/`, `.cache/` and `.venv/`. Cloning alone cannot restore them. Stop training and comparisons before copying so the run manifest, summaries, policies and latest pointer form one consistent snapshot. Back up both source and destination artifacts before restoration; never overwrite an existing run with the same UUID.

### What each artifact means

| Artifact | Purpose and sharing boundary |
| --- | --- |
| `learning/latest.json` | Selects the active run UUID. It is not the checkpoint itself. |
| `learning/<run UUID>/run.json` and `seed-<seed>/summary.json` | Configuration, source/runtime identity, counters and training history, including diagnostics and world records. These are reviewable training evidence, not model weights. |
| `learning/<run UUID>/evaluation.json` | Saved frozen comparison, policy hashes, failures and gate result, if evaluation was explicitly run. Reviewable evidence; absence means no such saved comparison. |
| `seed-<seed>/initial-policy.json` and `policy-*.json` | Actual actor weights plus inference metadata. Required for JavaScript inference/initial-versus-trained comparisons, not evidence-only exports. |
| `seed-<seed>/checkpoint.pt` | Actor/critic weights, optimizer, RNG states, configuration, identity, counters and history. Required to resume; share only as a trusted model/checkpoint transfer. |
| `roboproof/runs/motion/readiness*.json` | Environment checks and retained replay reports. Shareable after review, but not learned-improvement evidence. |
| `roboproof/runs/motion/sessions/` | Saved motion experiments/replays, separate from training history. Transfer separately if wanted. |
| `.cache/robotai-nemotron/*.gguf` and pinned runtime files | Pretrained local conversation model/runtime, unrelated to the PPO training checkpoint. Large model assets; review redistribution terms separately. |
| `.cache/robotai-nemotron/local-api-key.txt`, API-key environment values | Private credentials. Never include in evidence, public archives, commits, browser URLs or chat. Generate a new local key on the target. |
| `roboproof/runs/nemotron/`, `roboproof/runs/nemotron-chat/` and runtime logs | Potentially private prompts, conversations and tool sessions. Not motion-training checkpoints; review before any sharing. |

For evidence-only sharing, select `run.json`, each `summary.json`, existing `evaluation.json` and relevant readiness reports, with the source snapshot identity and runtime versions. Inspect for private content first. Do not send the entire learning folder as an evidence-only export: even JSON policy files contain weights. Hashes identify artifacts but do not replace the artifacts required for inference or resumption. Model weights are not API secrets, and possession of weights alone does not establish training quality.

### Trusted full learning transfer

On the source computer, from the repository root, after stopping all writers:

```powershell
git rev-parse HEAD
git status --short
node --version
& '.\roboproof\gpu\.venv\Scripts\python.exe' -c "import sys, torch; print(sys.version); print(torch.__version__)"
Compress-Archive -LiteralPath .\roboproof\runs\motion\learning -DestinationPath ..\Nationals-motion-learning.zip
Get-FileHash -LiteralPath ..\Nationals-motion-learning.zip -Algorithm SHA256
```

Retain the printed source/runtime information and checksum with your private transfer record. Use a new archive filename if it already exists. This archive includes checkpoint and actor weights, not just shareable evidence. It deliberately excludes `.venv`, Nemotron weights, keys and other run directories. Obtain any reviewed uncommitted source files separately; the archive is not a source snapshot.

On the target, create the environment afresh and use the same reviewed source/WASM pair and exact runtime versions. Place the trusted archive one folder above the clone, compare its SHA-256 with the source record, then restore only into an absent learning directory:

```powershell
Get-FileHash -LiteralPath ..\Nationals-motion-learning.zip -Algorithm SHA256
if (Test-Path -LiteralPath .\roboproof\runs\motion\learning) { throw 'Existing learning history: back it up and choose a separate restore location.' }
New-Item -ItemType Directory -Path .\roboproof\runs\motion -Force | Out-Null
Expand-Archive -LiteralPath ..\Nationals-motion-learning.zip -DestinationPath .\roboproof\runs\motion
```

The archive's top-level `learning` folder restores `latest.json` and all UUID/seed subfolders in their original layout. For a non-default location, use the trainer's `--directory` and comparison CLI's `--directory` consistently. The standard dashboard reads the default location; it has no command-line training-directory override.

The dashboard's **Refresh training history** reads restored JSON actors/history without Python. Python needs the trusted `.pt` checkpoint to resume; JSON actor exports alone cannot restore critic, optimizer or RNG state. Do not treat downloaded arbitrary checkpoints as a public upload workflow. `.tmp` files are not committed checkpoints and must not be promoted over their completed counterparts.

### Resume a compatible run

For a run originally created with the example settings, replace the placeholder with its UUID:

```powershell
$runId = 'REPLACE-WITH-SAVED-RUN-UUID'
& '.\roboproof\gpu\.venv\Scripts\python.exe' -m roboproof.motion_learning.train --resume $runId --seed 42 --seed-count 3 --updates 8 --max-steps 4096 --rollout-steps 128 --max-seconds 180
```

Use the saved `run.json` settings, not these numbers, for differently configured runs. Seed, seed count, rollout settings, algorithm and curriculum must match. Update/step budgets may increase explicitly but cannot decrease. Source hashes, exact engine/WASM identity and exact PyTorch version must match; engine identity includes Node/V8/platform/architecture. Moving across Node versions or operating systems is not a promised compatible resume or byte-identical replay.

Resume returns to the last complete rollout/episode checkpoint, restoring RNG/optimizer state and discarding uncommitted work. It is not arbitrary mid-episode restoration. If the run has already reached the saved update budget, unchanged budgets do not request more updates. Preserve prior evidence before extending budgets. A changed trainer, curriculum, simulator or controller should normally start a new run, not bypass identity checks. In particular, the current `mixed-reach-v2` trainer cannot resume checkpoints from the older trainer-source identity.

## 5. Optional local Nemotron, separate from CPU training

Skip this section unless local conversation/task advice is wanted. It is not required for the simulator or motion learner. Setup makes explicit downloads; start launches a background CPU model server:

```powershell
.\tools\setup-nemotron.ps1
.\tools\start-nemotron.ps1 -Threads 4 -CacheMiB 128
node roboproof/server.js 8766
```

Do not start a second dashboard if one already owns port 8766. Start/restart the dashboard after the model launcher creates its key, then open [RobotAI](http://127.0.0.1:8766/#nemotron) and use **Check connection**. Availability is not measured inference or motion evidence.

`setup-nemotron.ps1` requires `curl.exe`, PowerShell archive support and access to the official artifact URLs pinned in `roboproof/nemotron-runtime.json`. It resumes `.part` downloads and verifies SHA-256/model size. `start-nemotron.ps1` verifies the model, original runtime ZIP and extracted runtime files, then creates `local-api-key.txt` if absent. For an offline trusted transfer, retain the manifest-matching model, runtime ZIP and extracted runtime directory, not only `llama-server.exe`; exclude the source machine's key and logs.

The launcher binds to `127.0.0.1:8080`, uses alias `robotai-nemotron` and explicitly passes `-ngl 0`: inference is CPU-only. `-CacheMiB 0` disables prompt caching, but neither it nor the default 128 MiB cap limits total process RAM. The multi-GB model can exhaust a 6 GB host alongside training. Closing the browser does not stop it; stop only the model process whose PID the launcher prints. Neither server starts automatically after reboot.

An AMD/NVIDIA adapter, a model filename, or installing this CPU requirements file does not enable GPU training. The canonical PPO trainer has no GPU device option. Separate tensor-backend ROCm validation requires a supported HIP PyTorch environment and real measured results; see [GPU boundaries](../roboproof/gpu/README.md). No GPU, cloud or physical result is implied by this guide.

## 6. Troubleshooting without weakening checks

| Symptom | Action |
| --- | --- |
| `node`/`python` not found, or Python opens the Store | Verify `Get-Command node` and `Get-Command python`, correct interpreter installation/PATH, and open a new terminal. Use `py -3.12` for base setup if available. The training bridge still needs Node on PATH or an intentional `ROBOTAI_NODE`. |
| `No module named torch` or no matching CPU wheel | Confirm Python 3.12 x64 and use the exact `.venv` executable for pip and training. Check the package-install error/network/disk availability; do not silently substitute a different Torch version or count skipped tests as passes. |
| `No module named roboproof` | Return to the repository root and run the documented `-m roboproof.motion_learning.train` command. |
| PowerShell blocks `.ps1` scripts | Use the direct Python HTTP-server command where applicable. Follow your organization's policy for reviewed scripts; do not change machine-wide execution policy as a setup shortcut. |
| Port already in use / `EADDRINUSE` | Stop the specific server you started, not all Node/Python processes. Alternatively use `node roboproof/server.js 8767` or `start-simulator.ps1 -Port 8768`, and open that URL. |
| WASM does not load or 3D is blank | Use HTTP, check that both WASM/build manifest exist, inspect browser errors, and check CDN access for 3D. Do not confuse a CDN failure with learner/controller validation. |
| `Stale production controller source` / `Stale production control.wasm` | Restore a matching reviewed source/WASM/build-manifest pair. If deliberately rebuilding changed C++ sources, the build script requires an explicit Zig executable; the supplied manifest records Zig 0.14.1. A relative invocation is `.\tools\build-wasm.ps1 -Zig .\.cache\toolchains\zig\zig.exe` if you have placed that tool there. Rebuilding changes artifacts; preserve existing work and rerun appropriate validation. Never edit hashes to suppress failures. |
| Empty/stale history, model hash mismatch, resume mismatch | Check `learning/latest.json`, its UUID folder, all seed policy/checkpoint files and their original hashes. Match source, Node/V8/platform/architecture and Torch identities. Evidence-only files cannot resume. Keep incompatible runs as historical evidence and start a new run rather than altering manifests/checkpoints. |
| `budget-stopped`, zero updates or no learned improvement | Inspect each seed's committed updates/history and discarded-step counts. Stop competing processes and decide explicitly whether to extend compatible budgets. Preserve negative evidence; readiness, runtime completion and the frozen improvement gate are different checks. |
| Nemotron checksum mismatch | Inspect the pinned manifest and transferred/downloaded file; setup refuses mismatched existing files. Quarantine the suspect file separately and reacquire a reviewed matching artifact. Never change the expected checksum just to launch it. |
| Nemotron unavailable, HTTP 401 or timeout | Check `.cache/robotai-nemotron/server.stderr.log`, its PID/port and model loading. Start/restart RoboProof after key creation. If changing `-Port`, set `ROBOTAI_NEMOTRON_BASE_URL` to the corresponding loopback `/v1` URL; automatic key-file reading only applies to the default `http://127.0.0.1:8080/v1`, so configure the newly generated local key through server-side `ROBOTAI_NEMOTRON_API_KEY`, never in a browser/prompt. Stop concurrent heavy work; a timeout is not successful inference. |
| Interrupted or apparently stuck training | Prefer Ctrl+C so owned bridge cleanup can run. After force-killing Python, identify and stop only its orphaned bridge child if one remains. Resume only completed checkpoints; do not rename temporary files or invent missing history. |

## 7. Recommended serialized validation

After reviewing setup, run focused checks one at a time with Nemotron stopped. The Python learning test below executes real gradients/resumption; it is not a static documentation check. Leave full suites, frozen evaluations and larger runs to the integration owner on a memory-constrained shared host.

```powershell
node tools/generate-config.js --check
node roboproof/cli.js motion-check
& '.\roboproof\gpu\.venv\Scripts\python.exe' tests/roboproof_motion_env_test.py -v
& '.\roboproof\gpu\.venv\Scripts\python.exe' tests/roboproof_motion_learning_test.py -v
node --test --test-concurrency=1 tests/motion-learner.test.js tests/motion-policy.test.js tests/motion-preparation.test.js tests/motion-api.test.js
```

Record actual exit codes, failures, skip reasons and evidence paths. No validation result is asserted here. See [episode contract](MOTION_EPISODE_CONTRACT.md), [motion learner](../roboproof/motion_learning/README.md) and [local Nemotron](../roboproof/NEMOTRON.md) for subsystem-specific constraints.
