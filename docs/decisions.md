# RoboProof decisions

## 2026-09-30 — Prepare real deep ML without disguising legacy evidence

- Decision: ship an isolated ML data/training/acquisition/confirmation pipeline and provider-neutral allowlisted job bundle after the user explicitly authorized project implementation. Deep learning is required by the revised plan.
- Reason: meaningful local contracts and actual gradient/resume tests can be completed before AMD API access; tensorized physics alone was not ML training.
- Alternatives considered: wait for credentials; call a tensor batch a trained model; pool legacy/canonical data; guess remote endpoints or deploy a foundation model.
- Risks: small legacy datasets are smoke evidence only, coarse duplicate bins do not establish region holdouts, probabilities/entropy are not safety guarantees, and local final-consumption markers are audit guards rather than access control.
- Affected components: `roboproof/ml/`, explorer ML evidence, readiness tests and roadmap. Canonical simulator work takes precedence over claiming integrated completion; no production library/controller changes or remote spending occur in this slice.

## 2026-09-28 — Preserve the existing active control core

- Decision: wrap `Cascade.hpp` through the existing hash-checked WASM binary.
- Reason: this executes real production controller source without requiring PROS hardware or rewriting the library.
- Alternatives considered: replace it with a toy PID; execute the complete PROS runtime; port first and validate later.
- Risks: odometry integration, scheduler, safety leases and hardware interfaces remain outside the shared binary.
- Affected components: controller adapter, scalar simulation, parity tests, ingestion.

## 2026-09-28 — Separate scalar model from browser game physics

- Decision: add a reduced-order headless model, after auditing the existing browser model; retain the browser model unchanged.
- Reason: explicit complete worlds, independent sensor timing/noise and a tractable scalar-to-tensor contract are needed for reproducible batch experiments.
- Alternatives considered: execute the whole DOM/game simulator in a VM; rewrite the browser physics; use instantaneous kinematics.
- Risks: two approximate plants with different constants and contracts; neither is physically calibrated. Their outcomes must not be presented as equivalent.
- Affected components: `sim.js`, `gpu/simulation.py`, scenario specification, metrics and limitations.

## 2026-09-28 — Start with CPU evidence and seed-local randomness

- Decision: source-checked WASM is the oracle; tensor controller/plant must pass CPU parity before claiming GPU verification.
- Reason: determinism and semantics are independently testable even when an AMD runtime is unavailable.
- Alternatives considered: GPU-only implementation; shared global random generator; invented GPU performance estimates.
- Risks: platform/backend floating-point differences require tolerances. Current tensor stepping may be slower on small CPU batches.
- Affected components: provenance, random streams, replay, CPU/Torch parity and benchmarks.

## 2026-09-28 — Keep fix proposals isolated and permit rejection

- Decision: evaluate time-scale configuration candidates only, using unchanged deadlines and gates, training-only selection, unseen holdout and counterexample regressions. Do not apply changes to firmware.
- Reason: a measurable regression report matters more than a cosmetically positive demo.
- Alternatives considered: automatically modify production gains; select on holdout; relax pass thresholds; cherry-pick scenarios.
- Risks: this candidate family may not fix the discovered structural localization failures. A higher mean success rate can still fail the zero-regression gate.
- Affected components: candidate selection, regression accounting, dashboard, exported proposal.

## 2026-09-29 — Make the model integration explicit

- Decision: default to a labeled deterministic tool investigation and provide an opt-in executable JSON bridge for a real model.
- Reason: no model endpoint or credentials were supplied; rule-based output must not masquerade as LLM reasoning.
- Alternatives considered: silently send repository source to a provider; fabricate a live AI trace; require a single provider SDK.
- Risks: bridge execution and any provider data transfer are trusted user configuration; model hypotheses are not proofs. Live LLM mode requires separate validation.
- Affected components: `agent.js`, evidence tools, source exposure, runtime/turn/simulation budgets.
