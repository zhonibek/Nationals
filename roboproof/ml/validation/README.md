# Retained local ML readiness evidence

`smoke.json` is a compact record of the **actual** 2026-09-30 CPU run, not a model, benchmark promise or fabricated fixture. The full dataset/package/model/checkpoint/card/selection/confirmation and offline AMD job bundle remain under ignored `roboproof/runs/ml-demo/` on the development machine. It is intentionally not an independent substitute for those artifacts.

Reproduce with the fresh-corpus commands in [ML instructions](../README.md), using a new output directory if the example paths already exist. Byte-identical serialized model archives are not promised across paths/platforms/PyTorch versions; compare recorded runtime/source identity, splits, learned tensors and numeric outputs explicitly. The same-runtime repeat/resume tests compare tensors, not archive filenames.

The final evaluation set was not consumed. The 8/8 selected-failure result is a small biased cohort with two random-exploration selections, **not evidence that ML beats random/evolutionary search**. No remote job, GPU training, live LLM request or physical test ran. Future canonical/AMD evidence must be recorded separately rather than relabeling this legacy CPU run.
