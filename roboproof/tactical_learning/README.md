# RoboProof Tactical AI pilot

This package starts a score-driven tactical policy for V5RC Override. It is
separate from Nemotron's read-only rule commentary and from the GVF/LTV-LQR/PID
motion stack.

## What the policy learns

At each decision, the environment builds a bounded set of legal high-level
candidate tasks, such as scoring a Pin or Cup in a compatible goal, **scoring a
match load from the alliance loading zone** (`load-score-pin` / `load-score-cup`),
changing a Toggle, or moving a robot into the midfield. A small actor-critic network ranks
the candidates. Its reward is the change in the original `OverrideGame` score
differential, not an LLM-generated score or a hand-entered point estimate.
The game rules engine validates each selected macro-action.

The current pilot deliberately uses an **abstract movement model**: a macro
action relocates a robot to its target and charges travel time using a
configurable nominal speed (24 in/s by default). It does not simulate a path,
collisions, odometry, GVF, LTV-LQR, PID, or the robot's physical pickup and
placement constraints. Thus its checkpoints can teach game-score priorities in
this abstract environment, but cannot establish a physically feasible tactic,
improved motion, a match win, or a deployment-ready policy. Actions are never
dispatched to a live robot.

## Remote training only

Training intentionally refuses CPU, ROCm and non-Linux runtimes, and any GPU
other than an approved free T4 or a separately approved Nebius L40S (below).
For the T4, run only in the approved private Colab T4 notebook after
transferring the reviewed project sources. A bounded pilot command is:

```bash
python -m roboproof.tactical_learning.train \
  --approved-free-t4 \
  --episodes 100 \
  --max-seconds 300 \
  --evaluation-episodes 12 \
  --output roboproof/runs/tactical/t4-pilot-<unique-id>
```

The output path must not already exist. The worker writes `run.json`,
`episodes.jsonl`, `exposure.jsonl`, periodic `latest.pt` checkpoints,
`training.jsonl`, and `evaluation.json`. The final `run.json` is written
before the Node bridge is shut down; its `bridgeShutdown` field records how
the bridge stopped (`exited`, `terminated`, `killed` or `kill-unconfirmed`),
and a stuck bridge does not change the trainer's exit status. Preserve the complete run folder,
hash it, and copy it to the approved private RoboProof Drive folder before
disconnecting the ephemeral runtime. Model weights and run artifacts are not
committed to Git.

### Optional warm start from the team tactic (off by default)

`--warm-start-demonstration-episodes N` (0–200; the default 0 runs the trainer
exactly as before) starts the fit from the `loader-cycle` tactic instead of
from scratch:

1. It plays N demonstration episodes. At each decision, the bridge's
   `demonstrate` operation returns the candidate `loader-cycle` would pick,
   the same choice `playMatch` makes. Demonstration seeds are the block right
   after the training seeds, `seed + episodes + i`, red for even `i` and blue
   for odd `i`. Each episode is logged to `exposure.jsonl` with
   `phase: demonstration` and to `warm-start.jsonl`.
2. It pretrains the actor with cross-entropy on the demonstrated choices and
   the critic with smooth-L1 on their discounted returns, using the RL
   discount. Pretraining has its own Adam optimizer (learning rate 1e-3,
   batch 256, `--warm-start-epochs`, default 200), so the RL optimizer starts
   fresh. The weights are saved as `warm-start.pt`.
3. The unchanged RL loop then runs from those weights.

The warm start shares `--max-seconds`. No demonstration starts after 40 % of
the budget, and pretraining stops at 50 %. With a warm start, RL also stops
early enough to leave twice the slowest RL episode for each evaluation
episode, so the evaluation still runs. The baselines after the evaluation are
outside the budget, as before. `run.json`, every checkpoint and
`evaluation.json` record the warm start: episodes, seeds, losses and
imitation accuracy. A warm-started result is therefore not a from-scratch
comparison with `loader-cycle`. To keep demonstrations cheap, `loader-cycle`
first previews only the match-load candidates and previews the rest only when
no load scores; the choice is identical.

### Nebius AI Cloud Serverless AI job (paid, separately approved; shelved)

**Shelved on 2026-10-24.** The user spends no money and the free Colab T4 is
available again, so seed 731201 is for the T4 fit. The launcher exits
immediately. Using it again needs a new unused seed in the job script, free
compute credits and the user's approval.

Because the free Colab T4 was not allocated, the same trainer has a second,
separately named opt-in: `--approved-nebius-gpu L40S --approved-max-cost-usd <usd>`.
It requires Linux, CUDA PyTorch, exactly one real NVIDIA L40S (capability 8.9,
at least 40 GiB) and Node.js, and records `provider: nebius-serverless-ai-job`,
the approved cost ceiling and `providerBillingVerified: false` (the trainer
cannot see billing). The T4 path is unchanged, and the two opt-ins are mutually
exclusive.

- [`tools/nebius-tactical-job.sh`](../../tools/nebius-tactical-job.sh) is the
  job entrypoint. It predeclares the one fit (seed 731201, 100 episodes, at most
  600 s, 8 evaluation episodes). It clones the public
  `nvidia-nebius-global-ai-hackathon` branch, pins the exact commit, installs
  Node.js 20.19.0 after checking its SHA-256, and runs the trainer. It always
  packages `run/` (`run.json`, `episodes.jsonl`, `exposure.jsonl`,
  `training.jsonl`, `evaluation.json`, `latest.pt`), `stdout.txt` (stdout and
  stderr), `nvidia-smi.txt` and `job.json`, including failed or partial runs. It
  writes a `SHA256SUMS` manifest and a `.tar.gz` with its own `.sha256`, then
  copies them to the private results bucket mounted at `/output`. The manifest
  and archive hash are also printed to the job log. It refuses to run if the
  bucket already has the seed's folder.
- [`tools/nebius-tactical-launch.sh`](../../tools/nebius-tactical-launch.sh)
  creates the job (`gpu-l40s-a`, `1gpu-8vcpu-32gb`, on-demand, 1 h timeout,
  50 GiB disk, PyTorch 2.8.0/CUDA 12.8 image pinned by digest). It only runs
  `--dry-run` unless you pass `--launch`, and it refuses a commit that is not on
  `origin/nvidia-nebius-global-ai-hackathon` or an approved cost below the
  timeout ceiling (about $1.60). Only free bonus credits may pay for it.
  `--free-credit-balance-usd`, read from the Billing page, must cover the
  approved cost, because Nebius charges a card on file automatically.

Retrieve the archive from the bucket, using the console's pre-signed "Get link"
or the S3 API. Check its SHA-256 against the job log, then preserve it locally
and in the private RoboProof Drive folder.

`status: completed` means only that the configured pilot budget ran. Both
`learnedImprovementVerified` and `independentQualificationPerformed` remain
false until a separately predeclared comparison beats a non-learning tactical
baseline on held-out scenarios with the actual Simulator/iraLIB motion stack.

## Give the trained policy a task in the app

Export the actor once from a preserved run. This only reads the weights; it
never trains:

```bash
python -m roboproof.tactical_learning.export_policy --checkpoint <preserved run>/run/latest.pt
```

It writes `roboproof/runs/tactical/policy.json` (Git-ignored, since weights are
not committed) with the run ID, checkpoint SHA-256, training seed, warm-start
summary and the held-out evaluation of the same run. Set
`ROBOPROOF_TACTICAL_POLICY` to use another exported file.

RoboProof serves it as `POST /api/tactics/task` (`{task, seed, alliance}`), and
the Simulator has a **Give Tactical AI a task** panel. Type one task in English
or Russian, for example "score 3 match loads into the neutral goals, then
park", "забей 2 кубка с поля за 30 секунд", "набери 50 очков" or "припаркуй
роботов".

- `roboproof/tactical_task.js` turns the text into an allowlisted spec: object
  (Pins, Cups), source (match loads or field), goals (neutral, alliance, tall),
  a number of placements, a points target, a time limit and parking. A task it
  does not understand, or a number out of range, is refused instead of guessed.
- The policy then plays the driver period choosing only among candidate
  macro-actions that fit the task, always the highest-scoring one, as in the
  held-out evaluation. Rejected actions are not retried in the same state.
  After the scoring part it parks if asked. The rules engine scores each step,
  and the response lists every decision, the placements, the own points gained
  and whether the task was completed.
- `roboproof/tactical_policy.js` runs the actor in plain JavaScript
  (tanh, tanh, linear). Its scores match PyTorch to about 1e-6.
- This is the actor-critic trained here (policy gradient with a learned critic
  baseline), not PPO. It acts in the abstract macro-action model only: nothing
  drives a robot, and a completed task is not a physical or scoring
  qualification.

## Loading-zone cycles, baselines and the Nemotron coordinator

- `load-score-*` actions drive to the alliance loader, request a match load
  through `OverrideGame.interact(..., 'load')`, pick it up at the outlet, carry
  it to a legal goal and place it. Feature 23 marks loader actions; the model ID
  is `override-tactical-actor-critic-v2-loader`.
- `TacticalEnvironment.previewCandidates()` returns every candidate's exact
  engine-previewed points and seconds (read-only, on a cloned game).
- Timing profiles: `measured` (default) is calibrated to the team's robot
  (about 2 s per loading-zone cycle, about 4 s per field cycle; 48.1 in/s from
  `config/robot.json`, pickups fitted by `calibrateTiming()` to this field
  geometry). `nominal` keeps the earlier uncalibrated assumptions. Individual
  times can still be overridden:
  `new TacticalEnvironment({timing, actionSeconds: {pickup, loaderPickup, load, place, toggle, park}})`.
  The profile name is part of every observation and of `episodes.jsonl`.
- [`../tactical_baselines.js`](../tactical_baselines.js) provides the
  non-learning tactics `loader-cycle` (team tactic), `field-greedy`,
  `greedy-all` and `random`, and one match runner shared with Nemotron.
- [`../tactical-coordinator.js`](../tactical-coordinator.js) lets Nemotron on
  Nebius Token Factory pick one offered candidate per decision through a
  `choose_action` tool call. Anything else fails closed and keeps the decisions
  made so far. The simulator's **Tactical AI match** panel and
  `POST /api/tactics/match` run the same code.
- The trainer's holdout evaluation now also plays the baselines on the same
  seeds and records `beatsLoaderCycleOnHoldout`. That flag is a development
  comparison, not independent qualification.

Measured results (seeds 101–102, red alliance) are in
[HACKATHON_POST.md](../../HACKATHON_POST.md#tactical-ai-results).
