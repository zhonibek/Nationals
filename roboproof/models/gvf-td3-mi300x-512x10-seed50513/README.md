# Experimental GVF-TD3 512x10 actor — seed 50513

This folder contains one trained actor snapshot from the AMD MI300X experiment. It is a research artifact, not a validated or recommended default policy.

## Recorded run

- Algorithm: `td3-tanh-gvf-residual-v1`
- Actor: width 512, 10 hidden layers
- Seed: 50513
- Training budget: 100,051 decisions, 564 episodes, 4,488 optimizer updates
- Runtime: approximately 1,838 seconds on AMD Instinct MI300X
- Result: `learnedImprovementVerified=false`; `independentQualificationPerformed=false`

The experiment did not establish improved movement, independent qualification, physical-robot safety, or permission to replace the existing controller. The full optimizer checkpoint is not included in this repository.

## File and integrity

`actor-episode-000564.json.gz` is the final actor JSON compressed with gzip.

- Compressed SHA-256: `d8a18ea0b8c3af12541b80724f75d822c4a711b94ad471de65bf0221670fef9a`
- Decompressed JSON SHA-256: `cc6b36a262c3e90b6a8d033989c39287ad0a3f3150fefef687c57434bea8d4bf`

Decompress with Node.js:

```sh
node -e "const fs=require('node:fs'),z=require('node:zlib');fs.writeFileSync('actor-episode-000564.json',z.gunzipSync(fs.readFileSync('actor-episode-000564.json.gz')))"
```

The resulting JSON is the actor snapshot format consumed by RoboProof's experimental motion-learning tools. It is not a complete resumable training checkpoint.
