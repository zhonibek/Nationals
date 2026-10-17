# Using a saved GVF actor

## What is and is not on GitHub

The repository contains the GVF/TD3 implementation and its tests, not trained model weights. Training outputs under `roboproof/runs/` and private custody/cache files are excluded from Git. This is intentional: checkpoints include optimizer and replay state, can be large, and are not source code. Do not commit an unreviewed run directory or a checkpoint to make a model appear deployable.

For inference, the relevant artifact is the exported actor JSON referenced by the completed run's `run.json`, normally `actors/actor-episode-<number>.json`. A `checkpoint.pt` is for exact training resumption; it is not the deployment actor. Transfer the actor from its private archive or approved server to the computer that has the matching project source. Do not use an actor if its run is incomplete, its SHA-256 does not match the run manifest, or the original Simulator/controller identity check fails.

## Run one explicit simulator experiment

The checked-in inference entry point is `roboproof/motion-gvf-learner.js`. It validates the actor's schema, layer dimensions, normalization, SHA-256 and exact original Simulator/iraLIB identity, then runs one replayable motion task in the original simulator. It does not train, promote the actor, or certify improvement. The dashboard currently has no saved-GVF-actor import/activation control; the command below is an explicit CLI experiment, not automatic or physical-robot deployment.

In PowerShell, from the repository root, set `$actor` to the transferred exported actor JSON. Use the exact SHA-256 from the run's `run.json` as `$expectedSha256` (do not replace it with a newly calculated hash when checking a transfer):

```powershell
$actor = 'C:\path\to\actors\actor-episode-000562.json'
$expectedSha256 = 'PUT_THE_RUN_MANIFEST_SHA256_HERE'
node -e 'const result = require("./roboproof/motion-gvf-learner").runActor({ filename: process.argv[1], sha256: process.argv[2], seed: 4242, task: { start: { xIn: 0, yIn: 0, headingDeg: 0 }, goal: { xIn: 0, yIn: 24, headingDeg: 0 }, deadlineSeconds: 10 }, configuration: {} }); console.log(JSON.stringify({ report: result.report, inferencePerformed: result.inferencePerformed, exactReplayVerified: result.exactReplayVerified, learnedImprovementVerified: result.learnedImprovementVerified, independentQualificationClaimed: result.independentQualificationClaimed }, null, 2));' $actor $expectedSha256
```

The task above is a single nominal example, not a benchmark. The returned report includes the measured simulator outcome and exact replay status. A result may say `learnedImprovementVerified: false`; this is expected until the predeclared independent acceptance gate passes. A failed source/contract/hash check must be fixed by obtaining the matching reviewed source and intact artifact, never bypassed by editing metadata or weakening validation.

## Before any policy promotion

Run the complete predeclared development and untouched independent suites with the required seeds, controller baseline and unchanged acceptance criteria. Preserve every run, exposure, failure, model hash and evaluation report. Promote only if every required gate passes; otherwise keep the controller-only default. This experimental GVF actor is not approved for physical robot control, game-score optimization, obstacle avoidance or autonomous tactical execution.
