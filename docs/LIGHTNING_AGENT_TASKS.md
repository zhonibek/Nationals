# Bounded Lightning agent assignments

## Dispatch status — 2026-10-17

Prepared, **not dispatched**. The signed-in organization URL currently serves
Lightning's "October 2026 Security Incident" notice instead of the application.
The notice reports revoked access and unavailable platform services. Do not send
source code, authentication material or agent requests while this remains the
observed state. Do not try another endpoint or old saved credential as a bypass.
No model API key was retrieved or agent inference requested during this check.

Public introductory-token advertising is not an account quota receipt. Once
access is restored, verify the actual organization, eligible models, remaining
free token allowance, rate limits and no paid fallback before dispatch. Do not
purchase credits, add a payment method, enable top-up or allocate new compute.
Start with one bounded task per role, at most two requests concurrently and
2,000 output tokens per request; stop on uncertain billing or any quota error.

These are code-review/development agents, not robot actuators or training workers.
Only selected project source and compact simulated evidence may be sent. Never
send credential caches, tokens, personal files, untouched final cases or model
checkpoints. Each output is an untrusted proposal reviewed by the coordinator;
it does not authorize commands, source changes, training or deployment.

## Agent 1 — movement control review

Review the GVF v2 turn fix in `GuidingVectorField.hpp`, `Cascade.hpp`, the shared
hardware boundary and Simulator adapter. Check that the **combined** feedforward
and LQR angular request is bounded, precise completion remains sensor-only, and
the ordinary controller stays numerically unchanged. Identify concrete defects
with file/line evidence and propose a minimal patch. Do not tune constants from
consumed final outcomes, alter physics, evaluation gates or the default mode.

Deliverable: a bounded defect/patch proposal and missing focused test cases.
Write scope if subsequently approved: control-related test additions only;
production changes belong to the coordinator.

## Agent 2 — exposure recovery admission

Design and implement a standalone fail-closed reader for the independently
audited conservative exposure envelope. Bind the exact auditor, audit, archived
inputs, manifest, every shard, seed/ordinal ranges and metadata-only scope. Reject
missing, replaced, reordered, duplicated, symlinked or unstable inputs. Keep the
original loss event unresolved and all reserved seeds recorded. Do not count
recovery records twice or omit historical measured/development/consumed-final
exclusions. Do not infer missing models, actual query counts or pass verdicts.

Deliverable: proposed reader plus tiny synthetic positive/negative tests.
Write scope if approved: a new admission module and its dedicated tests only;
no registry clearing, existing collector change or full local envelope generation.

## Agent 3 — GVF residual-learning contract

Specify a distinct versioned observation/action contract for residual learning
over the actual shared GVF/C++ controller. Separate it from the existing four
actions `[pace, dX, dY, dTheta]`; do not relabel existing actors or datasets.
Define sensor-only inputs, physical units, bounded residuals, reward accounting,
reset/termination behavior, teacher provenance and all source/runtime bindings.
Explain what data and tests are missing before native teacher collection.

Deliverable: an actionable contract/design proposal, not a trained-model claim.
Write scope if approved: one new design document only. No training, model
downloads, reward implementation or old checkpoint conversion.

## Agent 4 — independent experiment verifier

Review the complete original-Simulator comparison procedure against the existing
controller and each candidate's own untrained policy. Require fixed budgets,
all declared seeds/cases/failures, unchanged gates and exposure-disjoint final
selection. Verify preservation before the next fit. Check that instrumentation
forwards every control argument, including the explicit GVF stepping mode.
Existing historical GVF worlds are development evidence, never a fresh final.

Deliverable: reproducible test commands and a requirement-to-evidence checklist.
Write scope if approved: dedicated evaluator tests only. Long tests, physics
campaigns and all optimization run only on a freshly verified approved remote
host, under a separately frozen bounded protocol. No local campaign or final draw.

## Agent 5 — tactics and product boundary

Map validated movement capabilities to a bounded future task coordinator:
reach-pose first, then separately validated pickup, placement and game scoring.
Keep rules grounded and distinguish advisory plans from executable skills.
Document missing gates and a simple chat-first UX flow. UX/voice/avatar work
remains plan-only; an agent answer is not evidence of working tactics, camera
navigation, collision avoidance, learned scoring or physical robot access.

Deliverable: prioritized integration checklist tied to existing modules.
Write scope if approved: one proposal document only; no frontend or tactical
execution changes before the relevant measured movement/skill gates pass.

## Coordinator acceptance

Keep request/model/source hashes, actual response/usage, quota observations and
failures without secrets. Never label these tasks running or completed before
actual submitted requests and returned results exist. Review and test proposed
patches separately; preserve old source/results and do not publish without an
explicit publish request. Before reusing Lightning, the account owner must handle
any credential rotation or revocation required by the provider's incident notice.
