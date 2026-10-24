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
- [~] 4.5 Train the tactical actor-critic with the new actions on the approved private Colab T4 (bounded foreground run per `AGENTS.md`), preserve outputs locally and in the private RoboProof Drive folder, evaluate on held-out seeds against `loader-cycle`.
  - **Plan: run on the free Colab T4 with seed 731201** (Colab offers a T4 again). The user spends no money. The Nebius route below is shelved and its launcher refuses to run.
  - Free Colab T4: not allocated at first (see log). Nebius route prepared: a separately named opt-in `--approved-nebius-gpu L40S --approved-max-cost-usd <usd>` in `train.py`, the job entrypoint `tools/nebius-tactical-job.sh` (seed 731201, 100 episodes, ≤600 s, 8 evaluation episodes; hashed archive copied to a private bucket) and `tools/nebius-tactical-launch.sh` (dry-run by default). The user approved pushing it and allows **only free bonus credits** to pay. **Not launched**: the launcher refuses unless the bonus-credit balance shown on the Billing page covers the approved ceiling.
- [x] 4.6 Document measured results honestly (README, `HACKATHON_POST.md`, `roboproof/tactical_learning/README.md`).

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
- Open for the tactics study: measured robot cycle times (loading-zone cycle vs field pickup) to calibrate `actionSeconds`; parallel robots in the tactical model.
