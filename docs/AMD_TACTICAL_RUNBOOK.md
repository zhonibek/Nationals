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

Default campaign: 34 parallel runs (the most that fit in GPU memory, see Progress), PPO, v3 features, 1024 × 10, 32-episode loader-cycle warm start, up to 3,000 episodes or 2 h of training per run, then held-out evaluation (8 start poses) against loader-cycle, field greedy, greedy-all and random. Seeds from 1400001 (stride 5,000); the completed launch used `--runs 34 --base-seed 1400001` (see Progress). Wall-clock limit 3 h; worst case **5.97 USD** at 1.99 USD/h.

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
