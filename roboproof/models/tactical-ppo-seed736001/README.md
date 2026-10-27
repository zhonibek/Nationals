# Tactical PPO policy — seed 736001

The trained Tactical AI policy that the app's **Tactical PPO optimizer** uses by default. It is the JSON export of the PyTorch checkpoint (`export_policy.py`), so the RoboProof server can run it without PyTorch.

- Model: `override-tactical-actor-critic-v2-loader`, 23 features per candidate action → 2 × 128 (tanh), 28,034 parameters.
- Training: behavior-cloning warm start from the `loader-cycle` tactic (18 matches), then 300 episodes of PPO (clipped surrogate, GAE) on the measured-timing profile.
- Hardware: NVIDIA Tesla T4 (free Google Colab), run ID `ec79ba4f-d5cf-4b7d-8665-8ea193e35ccd`, checkpoint SHA-256 `a39d4c68e65de168ab46486a101bcc13c7393d516bcd51ce067d1a4740bfa97b`. The complete run record is in `docs/progress.md`.
- Held-out result (8 start poses, abstract match model): mean margin 201.5, against loader-cycle 244.5, field greedy 217.4, greedy-all 203 and random 92. It does **not** beat the team's loading-zone tactic (`beatsLoaderCycleOnHoldout: false`).

A policy exported locally to `roboproof/runs/tactical/policy.json` (or named by `ROBOPROOF_TACTICAL_POLICY`) takes precedence over this file.
