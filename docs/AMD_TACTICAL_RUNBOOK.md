# Tactical AI on an AMD Instinct MI300X — plan, runbook and progress

Branch: `tactical-training` (the hackathon branch stays frozen until judging ends on 2026-12-15).
Provider: AMD Developer Cloud, one MI300X droplet, paid only from the free credit (USD 57.06 on 2026-11-06, expiring 2026-11-16).

## Goal

Train the Tactical AI (the policy behind **Tactical PPO optimizer · give Tactical AI a task**) until it beats the team's loading-zone tactic on held-out matches. Current best learned result: PPO seed 736001, mean margin 201.5 vs loader-cycle 244.5 (`beatsLoaderCycleOnHoldout: false`).

## What is prepared

| Piece | Where | Status |
| --- | --- | --- |
| 6.9× faster rules-engine previews, bit-identical results | `roboproof/tactical_environment.js` | done, verified on 12 full matches |
| v3 features: 7 rules-engine preview features per candidate (legal, points, seconds, points/s, rank, best overall, best match load) | `roboproof/tactical_environment.js`, bridge `--features v3` | done, tested |
| Model up to 1024 wide × 10 hidden layers (pre-norm residual blocks beyond 2 layers); the planned v3 1024 × 10 model has **10,022,914 parameters** | `roboproof/tactical_learning/model.py` | done, tested; the 2-layer v2 model is unchanged |
| Trainer options `--features`, `--width`, `--hidden-layers`, `--approved-amd-mi300x` (accepts only a real MI300X: ROCm build, gfx942, ≥180 GB; up to 6 h per fit) | `roboproof/tactical_learning/train.py` | done, tested with simulated runtimes |
| Campaign planner: predeclares every run, refuses seed overlaps (with `consumed_seeds.json` and between runs) and cost above the approved ceiling; summary; GPU utilization sampler | `roboproof/tactical_learning/campaign.py`, `consumed_seeds.json` | done, tested |
| Launcher: dry run by default; `--launch` checks the GPU, Node.js and a clean commit, starts all runs with a hard wall-clock kill, samples GPU use every 15 s, summarizes, hashes and packages | `tools/amd-tactical-campaign.sh` | done, dry-run checked locally |
| Independent check on fresh start poses: fixes the candidates (top runs by development margin, primary = best) before playing, refuses any seed a run trained, imitated or evaluated on, reports the paired margin difference against loader-cycle with a bootstrap 95% interval | `roboproof/tactical_learning/evaluate.py` | done, tested; checked on the MI300X with the probe run |
| Export of v3 / deep policies to the app: JSON manifest + float32 weights file (SHA-256 checked), residual forward pass in JavaScript, v3 features in the task runner and server | `export_policy.py`, `roboproof/tactical_policy.js`, `tactical_task.js`, `server.js` | done, tested against a separate reference; the PyTorch-vs-JavaScript check runs where PyTorch is |

Default campaign: 34 parallel runs (the most that fit in GPU memory, see Progress), PPO, v3 features, 1024 × 10, 32-episode loader-cycle warm start, up to 3,000 episodes or 2 h of training per run, then held-out evaluation (8 start poses) against loader-cycle, field greedy, greedy-all and random. Seeds from 2600001 (stride 5,000); the completed launch used `--runs 34 --base-seed 1400001` (see Progress). Wall-clock limit 3 h; worst case **5.97 USD** at 1.99 USD/h.

## Honest expectations about GPU use

Each decision needs the Node.js rules engine on a CPU core; the GPU only scores candidates and runs PPO updates. With 40 runs the GPU will be busy far more than with one, but how busy depends on the droplet's CPU cores. The campaign records GPU use (`rocm-smi`, or `amd-smi` `gfx_activity`) every 15 s and reports the measured mean and maximum in `summary.json`. This is the share of time the GPU had work queued, not the share of its arithmetic capacity in use: many small network evaluations keep it busy without loading it. A bigger model is compared with the small one on the same held-out seeds; bigger is not assumed to be better.

## Runbook

1. **Create the droplet.** AMD Developer Cloud → GPU Droplets → MI300X × 1 → an image with ROCm and PyTorch preinstalled → add your SSH key. Note the hourly price shown in the console.
2. **Check the machine** (SSH in):
   ```bash
   rocm-smi --showproductname
   python3 -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
   nproc; free -g
   ```
3. **Find PyTorch and add Node.js.** On the *PyTorch 2.14 (ROCm 10.1)* image, PyTorch runs inside a Docker container named `rocm`, not on the host, and the host folder `/shared-docker` is mounted into it. Clone onto `/shared-docker`, unpack the official Node.js binary there (check it against `SHASUMS256.txt`), link `node` into the container, install `git` inside it, and run every command below with `docker exec -w /shared-docker/Nationals rocm bash -lc "..."` from a host `tmux`. The container has `amd-smi`, not `rocm-smi`.
4. **Get the code** at the pinned commit:
   ```bash
   cd /shared-docker && git clone --branch tactical-training https://github.com/zhonibek/Nationals.git && cd Nationals
   git log -1 --format=%H
   ```
5. **Dry run** (plans and checks, starts nothing):
   ```bash
   tools/amd-tactical-campaign.sh --hourly-rate-usd <price from the console> --approved-max-cost-usd 20
   ```
6. **Smoke test** (one short run, about 15 minutes of droplet time):
   ```bash
   tools/amd-tactical-campaign.sh --launch --runs 1 --episodes 20 --max-seconds 300 --warm-start 4 \
     --hourly-rate-usd <price> --approved-max-cost-usd 2 --output-root roboproof/runs/tactical/mi300x-smoke
   ```
   Check `run-00.log`, `run-00/run.json` (`status`, `runtime.gpu`), and that `gpu-utilization.jsonl` has samples. The 2026-11-06 smoke run used seeds 750001–750024 (now in `consumed_seeds.json`); a new smoke run needs another `--base-seed`.
7. **Full campaign** inside tmux, so a dropped SSH connection does not stop it:
   ```bash
   tmux new -s campaign
   tools/amd-tactical-campaign.sh --launch --hourly-rate-usd <price> --approved-max-cost-usd 20
   ```
   Detach with `Ctrl+B D`; watch with `tail -f roboproof/runs/tactical/mi300x-campaign-*/run-00.log` and `amd-smi metric --mem-usage`. Each 1024 × 10 run peaks at about 5 GB of GPU memory during the warm start, so 34 runs is the safe maximum on one MI300X.
8. **Independent check** on fresh start poses (minutes; pick a `--base-seed` outside `consumed_seeds.json`):
   ```bash
   python3 -m roboproof.tactical_learning.evaluate --approved-amd-mi300x --campaign roboproof/runs/tactical/mi300x-campaign-34 \n     --top 5 --base-seed 2100001 --episodes 200 --output roboproof/runs/tactical/mi300x-campaign-34-independent
   ```
   To use a passing primary candidate in the app, export it next to the check (`python3 -m roboproof.tactical_learning.export_policy --checkpoint <run>/latest.pt --independent-holdout <check>/summary.json --output <folder>/policy.json`) and copy `policy.json` with `policy.weights.bin` to `roboproof/runs/tactical/` on the computer.
9. **Collect**: when the script prints the archive and its SHA-256, copy both to the computer (`scp`), check `sha256sum -c`, and upload them to the private RoboProof Drive folder.
10. **Destroy the droplet** in the console. A powered-off droplet can still be billed; only destroying it stops the charge.
11. **Record**: add the campaign's and the check's seed ranges to `roboproof/tactical_learning/consumed_seeds.json`, write the results (summary, best run, measured GPU use, cost) to `docs/progress.md`, and keep `learnedImprovementVerified` false until an independent check passes.

## Progress

- 2026-11-04 — Environment speed-up (6.9×, bit-identical), v3 preview features, configurable deep model (1024 × 10 = 10.0 M parameters), MI300X runtime guard, campaign planner/launcher, tests (Node 620 with 2 skipped, tactical Python 44). No training has run.
- 2026-11-06 — Droplet `roboproof-tactical-mi300x` (ATL1, MI300X VF gfx942, 191.7 GB VRAM, 20 vCPU, 235 GB RAM, USD 1.99/h, free credit only) created about 11:20 UTC. Pinned commit ec9122d cloned; Node v24.22.1 (checksum OK). Smoke run seed 750001 (20 PPO episodes + 4 demonstrations, 1024 × 10, v3): completed in 85 s, exit 0, held-out margin 250.625 vs loader-cycle 244.5 (development comparison only), archive SHA-256 `f94054492c99f9ad8c6be452efd543a83a61f98ac230f103e30128045153d932`. Found and fixed: the summary ignored amd-smi's nested `gfx_activity` values. Measured host memory per trainer: 4.1 GB with hipBLASLt, 1.8 GB with rocBLAS (`TORCH_BLAS_PREFER_HIPBLASLT=0`, now the launcher default), plus 0.22 GB per Node rules engine, so 64 runs fit in RAM.
- 2026-11-06 — First launch, 64 runs from seed 960001 (`mi300x-campaign-64`, archive SHA-256 `21304bd15bf313102a4173bf7e751d003726d1e7d5deac31fbe7b801ee534e34`): GPU memory ran out. 15 runs failed in warm start with `HIPBLAS_STATUS_ALLOC_FAILED` (192.9 of 196.3 GB in use); the other 49 were stopped with SIGTERM before any finished warm start (exit 143; their `run.json` still says `running` because the trainer does not record SIGTERM). All 64 seed ranges are registered as consumed. A memory probe (seed 1300001, 40 episodes, 32 demonstrations; archive SHA-256 `911bb9a784671cc8490a29c1cb19b2308ab4b4932cee8c7db877cd60e028a999`) measured the real per-run peak: **5.0 GB GPU memory** (during the full-batch warm start) and 2.3 GB host memory. GPU memory, not RAM, limits parallel runs: 34 runs use about 171 GB, leaving a 20 GB margin.
- 2026-11-06 — **34-run campaign** `mi300x-campaign-34` (commit 1307ff5, seeds 1400001–1568032, PPO, v3 features, 1024 × 10, 32-demonstration warm start, `TORCH_BLAS_PREFER_HIPBLASLT=0`), 11:49–13:50 UTC. All 34 runs exited 0 at the 2 h training budget (1,624–2,376 episodes each, median 2,180); no out-of-memory kills; GPU memory peaked at 166.7 GB; GPU busy (time share with work queued, not arithmetic load) 99.8 % over 475 samples; the CPU-side rules engine is the limit. Development holdout (each run's own 8 start poses): 25 of 34 runs beat loader-cycle (244.5); mean learned margin 257.5, best 303.5 (run 08, seed 1440001), worst 220.5. Archive 5,016,430,192 bytes, SHA-256 `f24018a0a34e0cf2e374c2bb640f9a0cb7f3a79733608460dbbf501334f36e05`; results-only bundle (no checkpoints) SHA-256 in `mi300x-campaign-34.results.tar.gz.sha256`.
- 2026-11-06 — **Independent check** (`evaluate.py`, commit 9058368 in a separate clone; its rules-engine source hashes match the runs') of the top 5 runs on 200 fresh start poses (seeds 2100001–2100200, alternating red/blue), candidates and primary fixed in `plan.json` before any episode. Every candidate beat loader-cycle on all 200 poses: run 08 (primary) 303.5 vs 244.5, paired difference +59.0, bootstrap 95 % interval [57.04, 60.96]; run 20 +53.5; run 14 +49.0; run 02 +48.2; run 27 +41.3. Other baselines on the same poses: field-greedy 220.2, greedy-all 203.0, random 87.5. Archive SHA-256 `f7a66b4cf47a4b20b9d31d2de4780e0289e56f6aceddf831cb7101a9c54cf408`. Start poses barely change outcomes in this abstract model (run 08 scores 291 as red and 316 as blue on every pose; loader-cycle 246 and 243), so the check confirms a better plan for this game model, not robustness to different situations. `learnedImprovementVerified` stays false: the gain is in the abstract macro-action model only and still needs the user's approval and a physical check.
- 2026-11-06 — Run 08 exported with `export_policy.py --independent-holdout` (manifest + 37,990,404-byte float32 weights). On the MI300X droplet the JavaScript runner reproduced PyTorch exactly on a real 1024 × 10 checkpoint (same margins and decisions on 4 matches); it needs about 10–15 ms per candidate there, about 30 s per match with 34 trainers running. Scoring all candidates together was measured slower and reverted.
- 2026-11-06 — **Multi-task Tactical AI (feature set v4).** A typed task becomes a task spec that `roboproof/tactical_task_environment.js` applies on top of the rules engine (allowed moves, phases, stop conditions, task reward); the app's task runner uses the same class and matched the old runner on 72 task runs. v4 adds a 22-value task block to the 30 v3 features (52 in total). Training samples one random task per episode seed (`roboproof/tactical_task_training.js`); a per-task greedy tactic (`task-greedy`) is the warm-start teacher and the comparison bar. `--parallel-envs K` steps K rules engines in lockstep with one batched network call (about 1.6× faster per trainer with K = 3 in the smoke runs).
- 2026-11-06 — **16-run v4 campaign** `mi300x-v4-campaign-16` (commit a42abdd, seeds 2300001–2490032, 16 runs × 3 rules engines, PPO, 1024 × 10, 32 task-greedy demonstrations), 15:13–16:37 UTC. Every run completed its 10,000 episodes (exit 0). On each run's own 64 held-out tasks, 15 of 16 beat task-greedy and 15 beat loader-cycle; best run 06 (seed 2372001) task return 94.5 vs task-greedy 83.6. Archive 2,398,471,879 bytes, SHA-256 `96fd650728314f52072a3b5b97cb95d25e1c985a7554f7833eca3e5c915d9b1c`; results-only bundle SHA-256 `a6f102ff6a7245b193c5a8aad9b198b1d94a1bda8c0a85201ec4c96768ce3fcd`.
- 2026-11-06 — **Independent check of the v4 campaign** on 200 fresh tasks (seeds 2500001–2500200, candidates and primary fixed beforehand). All five candidates beat task-greedy and loader-cycle on paired task return: primary run 06 80.4 vs task-greedy 72.0, +8.5 (bootstrap 95 % [6.4, 10.6]; 73 wins, 81 ties, 46 losses), success rate 81 % vs 80 %; run 03 +8.9, run 01 +7.9, run 07 +3.9, run 10 +6.7. Archive SHA-256 `35e062011403f4ccb80d4b5d8b5854f2736c2b563b577ab9bac2355e04fbc531`. The gain is mostly speed and points within tasks, not more completed tasks; it is for the abstract macro-action model only, and `learnedImprovementVerified` stays false pending the user's approval and a physical check. Run 06 is exported (manifest + 38,080,516-byte weights) with this result attached.
- 2026-11-06 — **Preservation and droplet end.** Large archives went from the droplet straight to the private RoboProof Drive with rclone, because the link to this computer ran at only 33–55 KB/s. Verified in `RoboProof/tactical-mi300x-20261106/` with `rclone check` (MD5, 0 differences): the full `mi300x-campaign-34.tar.gz` (5,016,430,192 bytes) with its `.sha256`, and `export-run08/` and `export-v4-run06/` (3 files each). Also uploaded there with matching sizes (not hash-checked in Drive): the smoke, memory-probe, campaign-64, probe-independent-check and v4 smoke K = 1 / K = 3 archives with their `.sha256` files, both results-only bundles, both independent-check archives and the earlier results-only bundles. On this computer (`roboproof/runs/tactical/mi300x-2026-11-06/`, git-ignored) `sha256sum -c` passes for both results-only bundles, both independent-check archives and both exports. **Lost:** the full `mi300x-v4-campaign-16.tar.gz` (2,398,471,879 bytes, SHA-256 `96fd650728314f52072a3b5b97cb95d25e1c985a7554f7833eca3e5c915d9b1c`). Google rate-limited rclone's shared client (`RATE_LIMIT_EXCEEDED`), and by the user's choice the droplet was destroyed after 30 more minutes of uploading; only parts 00–03 of 23 (100 MB each), `parts.sha256` and a README reached `mi300x-v4-campaign-16-parts/`. So no training checkpoint of the 16 v4 runs survives: run 06's actor survives as the exported policy, and all v4 results, logs and the independent check are kept. The rclone login was deleted on the droplet first. The droplet was destroyed at about 20:13 UTC (gone from the GPU Droplets list, SSH times out) after about 8.9 h: roughly USD 17.7 of the USD 57.06 free credit at USD 1.99/h, estimated from the hourly rate. Next time: give rclone its own Google OAuth client, and start copying each campaign archive as soon as it is written.
