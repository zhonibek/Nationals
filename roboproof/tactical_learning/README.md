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

Training intentionally refuses CPU, ROCm, non-Linux, and non-T4 runtimes. Run
only in the approved private Colab T4 notebook after transferring the reviewed
project sources. A bounded pilot command is:

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
`training.jsonl`, and `evaluation.json`. Preserve the complete run folder,
hash it, and copy it to the approved private RoboProof Drive folder before
disconnecting the ephemeral runtime. Model weights and run artifacts are not
committed to Git.

`status: completed` means only that the configured pilot budget ran. Both
`learnedImprovementVerified` and `independentQualificationPerformed` remain
false until a separately predeclared comparison beats a non-learning tactical
baseline on held-out scenarios with the actual Simulator/iraLIB motion stack.

## Loading-zone cycles, baselines and the Nemotron coordinator

- `load-score-*` actions drive to the alliance loader, request a match load
  through `OverrideGame.interact(..., 'load')`, pick it up at the outlet, carry
  it to a legal goal and place it. Feature 23 marks loader actions; the model ID
  is `override-tactical-actor-critic-v2-loader`.
- `TacticalEnvironment.previewCandidates()` returns every candidate's exact
  engine-previewed points and seconds (read-only, on a cloned game).
- Action times are configurable:
  `new TacticalEnvironment({actionSeconds: {pickup, loaderPickup, load, place, toggle, park}})`.
  Defaults are nominal; replace them with times measured on the robot.
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
