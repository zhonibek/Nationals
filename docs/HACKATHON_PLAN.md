# Nebius × NVIDIA Global AI Hackathon — plan and progress

Track: **Physical AI**. Deadline: **2026-10-30 10:00 PT**. Branch: `nvidia-nebius-global-ai-hackathon` (GitHub default branch).
Rules: <https://nebiusglobalaihackathon.devpost.com/rules>. This file is the single source of truth for what is planned, done and still open.

Status legend: `[x]` done and verified · `[~]` in progress · `[ ]` not started.

## 1. Branch and publication

- [x] Create `nvidia-nebius-global-ai-hackathon` from the previous hackathon working branch, including the verified Nebius Token Factory integration.
- [x] Push it and make it the default branch on GitHub, so judges see the hackathon code first.

## 2. English-only documentation

Also done: the simulator, portal and dashboard UI are in English; the natural-language command parser still accepts Russian input.

The rules require English submission materials (or translations).

- [x] Translate every README (`README.md`, `simulator/README.md`, all `roboproof/**/README.md`) to English.
- [x] Rewrite `HACKATHON_POST.md` in English as the Devpost text: problem, what we built during the submission period, how Nemotron + Token Factory are used, measured results, feedback on Nebius and NVIDIA.
- [x] Translate the remaining Markdown docs that contain Russian (`docs/**`, `.github/CONTRIBUTING.md`).
- [x] Replace `file:///C:/Users/...` links with repository-relative links.

## 3. Copyright and attribution

- [x] `LICENSE`: project copyright (MIT) for the original work, LemLib's MIT notice retained because LemLib code is included.
- [x] `THIRD_PARTY_NOTICES.md`: LemLib (MIT), Pedro Pathing (BSD-3-Clause; the original library is Java, our C++ implementation is an independent rewrite derived from its design, so its notice is retained), PROS kernel (MPL-2.0), Eigen (MPL-2.0), {fmt} (MIT), Three.js (MIT, loaded from CDN), NVIDIA Nemotron/Cosmos (used through APIs under NVIDIA model licenses, weights not redistributed), VEX Override rules (paraphrased subset).
- [x] Pedro attribution in `NOTICE.md` files next to the C++ code (the sources stay unchanged because their SHA-256 hashes are bound to `simulator/control-build.json` and RoboProof provenance).
- [x] README section "License, attribution and originality".

## 4. Tactical AI: loading-zone scoring

Domain insight from the team (tested on the real robot): **scoring match loads from the loading zone is the fastest scoring cycle**, because the loader outlet is close to the alliance and neutral goals and the robot does not have to search the field.

Current gap: `roboproof/tactical_environment.js` offers only field objects, Toggles and parking. The match-load action exists in the rules engine (`OverrideGame.interact(..., 'load')`) but the tactical layer cannot choose it, so neither the learned policy nor Nemotron can use the best tactic.

- [x] 4.1 Add `load-score-pin` / `load-score-cup` macro-actions: drive to the alliance loader, request a match load through the rules engine, pick it up at the outlet, carry it to a legal goal and place it. Scores come only from `OverrideGame.score()`.
- [x] 4.2 Add non-learning tactical baselines on fixed seeds: `loader-cycle` (team tactic), `field-greedy` (best points per second on field objects) and `random`. Report score margin, points per decision and seconds per scored point.
- [x] 4.3 Nemotron tactical coordinator (Nebius Token Factory): each decision, Nemotron reads a compact match state and the legal candidates (with engine-previewed points and seconds) and must choose one allowlisted candidate ID by tool call; a deterministic orchestrator executes it in the rules engine and replans. Fast model (`Nemotron-3_5-Lightning`) for per-decision choices. Invalid or missing choices fail closed. Save the full decision log.
- [x] 4.4 RoboProof API and simulator UI to run a Nemotron-coordinated match and show the decision log next to the baselines.
- [x] 4.5 Train the tactical actor-critic with the new actions on the approved private Colab T4 (bounded foreground run per `AGENTS.md`), preserve outputs locally and in the private RoboProof Drive folder, evaluate on held-out seeds against `loader-cycle`.
  - **Done on the free Colab T4 (seed 731201, 2026-10-24).** Mean held-out margin 102 vs loader-cycle 176.5 (field greedy 225.5), so it does **not** beat loader-cycle. The outputs are preserved locally and in the private RoboProof Drive folder. The trainer exited with status 1 after writing `evaluation.json`, because the Node bridge did not shut down within 10 s, so `run.json` still says `running`. See the log.
  - **Longer fit on the free Colab T4 (seed 731401, 1200 episodes, 2026-10-25)** with the fixed trainer at `4407b0a`. It exited 0 with `status: completed` and the bridge shut down cleanly. Held-out mean margin 99.5 vs loader-cycle 176.5 and field greedy 225.5, so it still does **not** beat loader-cycle, although the training margin rose from 74 to 117.5. The run is preserved locally and in the private RoboProof Drive folder.
  - **Fit on the measured timing profile (seed 732701, 500 episodes, 2026-10-26)** at `2a1588a`, exit 0, `status: completed`. Held-out mean margin 112 vs loader-cycle 244.5, field greedy 221.25, greedy-all 203, random 92.9, so it does **not** beat loader-cycle. The episode count was cut from the requested 1200 to 500 because evaluation shares the 600 s budget. Training then took only about 3 min, so 1200 would also have fit. The run is preserved locally and in the private RoboProof Drive folder.
  - **Warm-started fit (seed 734001, 17 `loader-cycle` demonstrations + 300 RL episodes, 2026-10-26)** at `8b1b24d`, exit 0, `status: completed`, bridge exited cleanly. Held-out mean margin 191.5 vs loader-cycle 244.5, field greedy 220.75, greedy-all 203, random 82.1, so it does **not** beat loader-cycle, but it is the best learned result so far (112 before). The gain comes from the warm start; the RL training margin stayed at about 181. The run is preserved locally and in the private RoboProof Drive folder.
  - The user spends no money. The Nebius route below is shelved and its launcher refuses to run.
  - Free Colab T4: not allocated at first (see log). Nebius route prepared: a separately named opt-in `--approved-nebius-gpu L40S --approved-max-cost-usd <usd>` in `train.py`, the job entrypoint `tools/nebius-tactical-job.sh` (seed 731201, 100 episodes, ≤600 s, 8 evaluation episodes; hashed archive copied to a private bucket) and `tools/nebius-tactical-launch.sh` (dry-run by default). The user approved pushing it and allows **only free bonus credits** to pay. **Not launched**: the launcher refuses unless the bonus-credit balance shown on the Billing page covers the approved ceiling.
- [x] 4.6 Document measured results honestly (README, `HACKATHON_POST.md`, `roboproof/tactical_learning/README.md`).
- [x] 4.7 "Give Tactical AI a task" in the app: a typed task (English or Russian) becomes an allowlisted spec, and the trained policy (seed 734001, actor-critic with a loader-cycle warm start, not PPO) carries it out in the abstract tactical model using only actions that fit the task. `POST /api/tactics/task`, a Simulator panel and an export script from checkpoint to JSON; the weights stay Git-ignored.

Out of scope for this deadline: physical dispatch of tactical decisions to the robot, perception-driven state, opponent modelling.

## 5. Submission package

- [x] Devpost text from `HACKATHON_POST.md`; track: Physical AI.
- [x] Feedback section on Nebius Token Factory and NVIDIA Nemotron.
- [x] "Built during the submission period" section (first commit 2026-09-17, after the 2026-08-26 start).
- [ ] Video (team-owned; the robot is tested continuously).

## Progress log

- 2026-10-24 — Nebius Token Factory integration completed and verified (see `docs/progress.md`). Plan written.
- 2026-10-24 — Branch `nvidia-nebius-global-ai-hackathon` created, pushed and set as the GitHub default branch (verified through the GitHub API).
- 2026-10-24 — All Markdown documentation and the simulator UI translated to English; LICENSE, THIRD_PARTY_NOTICES.md and Pedro NOTICE files added.
- 2026-10-24 — Tactical AI: loading-zone actions, candidate previews, configurable action times, four baselines, Nemotron coordinator, `/api/tactics/match` and the simulator panel. Measured on seeds 101–102: field greedy 228, loader-cycle 182, Nemotron Super 181, greedy-all 173, Nemotron Lightning 163, random 79 (mean margin). Node suite 594 tests: 592 pass, 2 skipped.
- 2026-10-24 — 4.5 attempted with the user's approval (Drive mount + result download): the approved notebook stayed in "provisioning" for about 6 minutes and Colab reported that compute units are exhausted and free GPU access is not guaranteed. No runtime was allocated, no code ran, no seeds were exposed (planned seed 731201 remains unused). Nothing was purchased.
- Options for 4.5: retry the free Colab T4 later, or run the same trainer as a Nebius AI Cloud Serverless Job (fits the Physical AI track; needs approved credits and its own preservation path).
- 2026-10-24 — 4.5 Nebius route prepared, not run. One on-demand `gpu-l40s-a` / `1gpu-8vcpu-32gb` Serverless AI job with a 1 h timeout: worst case about $1.55/h × 1 h + 50 GiB disk ≈ $1.60, billed per second. Runtime opt-in tests 6/6 and job-script tests 3/3 pass. No Nebius resource was created, nothing was purchased, and seed 731201 remains unused. Launch needs the user's approval in chat, the commit pushed to this branch, a private results bucket and the Nebius CLI.
- 2026-10-24 — The user approved pushing the Nebius route to this branch and set a constraint: only free bonus credits may pay, with no card charge, purchase or top-up. Nebius charges a card on file automatically once usage passes a threshold, so the launcher now requires `--free-credit-balance-usd` to be at least the approved ceiling and records it in `job.json`. Still not launched.
- 2026-10-24 — Nebius route shelved by the user: no money will be spent. Token Factory holds only $1 of credit and there are no compute credits, while the free Colab T4 is available again. 4.5 returns to the T4 with seed 731201. `tools/nebius-tactical-launch.sh` now exits immediately. Using it again needs a new unused seed, free compute credits and the user's approval. No Nebius job, bucket or other resource was ever created.
- 2026-10-24 — 4.5 ran on the free Colab Tesla T4 (driver 580.82.07, torch 2.11.0+cu130, Python 3.13, Node 20.19.0) at pinned commit `9b9ca8a`. The source hashes match the local commit. Seed 731201, 100 episodes, `--max-seconds 600`, 8 evaluation episodes. Training took about 30 s (13 updates); evaluation plus the baselines took about 7.5 min. Held-out mean margin: learned 102, loader-cycle 176.5, field greedy 225.5, greedy-all 174, random 44.25; `beatsLoaderCycleOnHoldout: false`. Trainer exit status 1: `NodeTacticalEnvironment.close()` timed out waiting for the bridge after `evaluation.json` was written, so `run.json` keeps `status: running`. No process was left behind. The archive `tactical-t4-seed731201.tar.gz` (325,462 bytes, SHA-256 `86dbe578619528389391cb05cbc9b740087d01beee89410d3cf21c09e6e4d5e6`) is saved locally and in `My Drive/RoboProof/tactical-t4-seed731201` (private, only you have access). `learnedImprovementVerified` and `independentQualificationPerformed` stay false.
- 2026-10-25 — 4.5 longer fit on the free Colab Tesla T4 (driver 580.82.07, torch 2.11.0+cu130), pinned at `4407b0a`; all 8 source hashes match. Seed 731401, 1200 episodes (exposure seeds 731401–732600), `--max-seconds 600`, 8 evaluation episodes. Training took 4.8 min (150 updates); evaluation plus the baselines took about 7 min. `status: completed`, trainer exit 0, bridge exited cleanly. Training margin per segment: 74.0 (episodes 1–100) → 117.5 (episodes 1101–1200). Held-out (seeds 1541062808–815): learned 99.5, loader-cycle 176.5, field greedy 225.5, greedy-all 175, random 58.6; `beatsLoaderCycleOnHoldout: false`. Archive `tactical-t4-seed731401.tar.gz` (351,734 bytes, SHA-256 `ad140ae8e2d628ca315bbd3d0f0a0217f3ddbac63be8bea226e0fc150e50873b`) saved locally and in `My Drive/RoboProof/tactical-t4-seed731401` (private).
- Open for the tactics study: measured robot cycle times (loading-zone cycle vs field pickup) to calibrate `actionSeconds`; parallel robots in the tactical model.
- 2026-10-25 — Team measurement on the robot: about 2 s per loading-zone cycle, about 4 s per field cycle. The tactical model now defaults to a `measured` timing profile (48.1 in/s from the drive configuration; match-load pickup 0.31 s, field pickup 2.67 s, place 0.4 s, fitted to this geometry). Seeds 101–102, mean margin: loader-cycle 246, Nemotron Lightning 231, field greedy 218, greedy-all 203, random 62.5; Nemotron Super 203 on seed 101 and failed closed on seed 102 (invalid tool call at decision 14). The learned-policy fits so far used nominal timing; the next fit should use the measured profile with a new unused seed.
- 2026-10-26 — 4.5 fit on the measured timing profile on the free Colab Tesla T4 (driver 580.82.07, torch 2.11.0+cu130), pinned at `2a1588a`; all 8 source hashes match. Seed 732701, 500 episodes (exposure seeds 732701–733200), `--max-seconds 600`, 8 evaluation episodes. Every episode records `timing: measured`. Training took about 3 min (63 updates); evaluation plus the baselines took about 9.5 min. `status: completed`, trainer exit 0, bridge exited cleanly. Training margin: 82.7 (episodes 1–50) → 119.9 (351–450) / 115.7 (451–500). Held-out (seeds 1541064584–591): learned 112 (red 108, blue 116), loader-cycle 244.5, field greedy 221.25, greedy-all 203, random 92.875; `beatsLoaderCycleOnHoldout: false`. Archive `tactical-t4-seed732701.tar.gz` (332,929 bytes, SHA-256 `ff4653096bfc597ec32d24eb42b75d128bd0dfb6ba4ec2e0aaa9d1fe15bab2d9`) saved locally and in `My Drive/RoboProof/tactical-t4-seed732701` (private).
- 2026-10-26 — 4.5 warm-started fit on the free Colab Tesla T4 (driver 580.82.07, torch 2.11.0+cu130), pinned at `8b1b24d`; all 8 source hashes match. Seed 734001, `--episodes 300 --warm-start-demonstration-episodes 32 --max-seconds 600 --evaluation-episodes 8`. A `loader-cycle` demonstration takes about 14.7 s on Colab, so the 240 s collection cutoff stopped at 17 of 32 (seeds 734301–734317; 734318–734332 were predeclared but never used and must not be reused as fresh worlds). Pretraining: 200 epochs on 476 decisions; imitation accuracy 1.9% → 61.3%. RL: 300 episodes (seeds 734001–734300), training margin 175–186 in every 50-episode segment. Held-out (seeds 1541069472–479): learned 191.5 (red 160, blue 223), loader-cycle 244.5, field greedy 220.75, greedy-all 203, random 82.1; `beatsLoaderCycleOnHoldout: false`. `status: completed`, trainer exit 0, bridge exited cleanly; 933 s including the baselines. Archive `tactical-t4-seed734001.tar.gz` (447,850 bytes, SHA-256 `96ff186e775708758d624fc95f30914c121bf8c80c395196691bab54aa1f3537`) saved locally and in `My Drive/RoboProof/tactical-t4-seed734001` (private).
