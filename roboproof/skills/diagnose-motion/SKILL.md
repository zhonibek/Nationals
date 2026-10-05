---
name: diagnose-motion
description: Explain a user-selected saved movement experiment using measured errors, settling gates and sampled telemetry. Use for failed motion, timeout, collision or result-analysis requests.
compatibility: Requires an explicitly selected local motion or Nemotron run; no new simulation.
allowed-tools: get_saved_motion_evidence finish_analysis
---
# Diagnose measured evidence

1. Call `get_saved_motion_evidence`. If no saved experiment was selected, ask the user to run and select one. Never invent a report, use a different session, or infer results from task text.
2. Quote the recorded termination reason and relevant metrics with units. Position alone is insufficient: completion also requires heading, linear speed, yaw rate and 15 consecutive settling ticks.
3. Describe the supplied final-state checks as final-state checks, not a reconstruction of the dwell interval. A time limit can occur even with a small endpoint error. Contact duration records simulator contact; it is not a real-world collision diagnosis.
4. Sensor and evaluator truth samples are sparse observations, not a complete time-series analysis. Missing telemetry means oscillation, saturation and localization causes remain unknown. Even visible correlations do not prove causation.
5. Call `finish_analysis`, separating measured facts, tentative explanations and a proposed next controlled experiment in the user's language. Do not change gains, extend the deadline automatically, rerun, train or claim an improvement from one successful example. If no selected report exists, call `ask_clarification` instead. Do not end with plain text.

The report hash binds this analysis to a saved snapshot. A stored exact-replay flag describes a previous check, not a fresh replay. Physical validation remains absent.
