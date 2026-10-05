---
name: verify-motion-improvement
description: Read saved PPO training and frozen baseline/untrained/learned evaluation. Use for has-it-learned, improvement, training progress and promotion-readiness questions.
compatibility: Read-only local checkpoint summary; does not train or rerun evaluation.
allowed-tools: get_learning_summary finish_analysis
---
# Verify improvement, not a favorable example

1. Call `get_learning_summary`. Distinguish missing, stale, running and completed evidence. Do not infer training or accepted improvement from the existence of a model name.
2. Distinguish changed actor weights from verified better movement. Report controller, untrained and learned success counts for every supplied seed, including failed worlds.
3. Preserve the full acceptance result. Promotion requires higher success than both comparators, zero success/contact regressions and at least 5% all-world mean time or effort-proxy gain across the required independent seeds. A single faster replay does not satisfy this gate.
4. If checkpoint hashes or corpus identity are stale, say the saved evaluation does not apply to the current checkpoint. Do not report its counts as a current comparison.
5. Call `finish_analysis` with the conclusion in the user's language. Do not select a favorable seed or tune against the inspected frozen corpus. Suggest training/development-only experiments and a fresh reserved final corpus. No automatic policy promotion, training, hardware or AMD-performance claim is permitted. Do not end with plain text.
