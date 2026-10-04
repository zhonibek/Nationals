# Measured Java differential evidence

Measured on 2026-09-29 with the unmodified surviving sibling Pedro checkout,
the repository's source-checked C++ WASM, and official Temurin JDK 8u504-b01.
The portable JDK archive was SHA-256 verified as
`ea43d46ede95b51e44a12c66711706cddc762e0a766c54bccea18954e902b2aa`.
No global Java installation or source restoration was performed.

`summary.json` preserves the actual source hashes, runtime, scope, tolerances,
per-output errors and the SHA-256 of the full measured report. Result: **PASS,
2,077/2,077 cases compared**, with no discrepancies outside the documented policy.
This is bounded cubic geometry validation, not whole-follower equivalence.

**177 low-speed curvature cases intentionally differ.** Java and C++ have
different singularity cutoffs. These cases remain in the full report with raw
results; the maximum raw curvature error is about 195,117.73. The policy-adjusted
maximum is about 0.00278104. It would be incorrect to describe all outputs as
numerically identical. The policy and fixed tolerances were defined before the
actual Java execution and were not relaxed to obtain this result.

Full inputs, compiled classes, raw Java/WASM results and every intentional
difference remain in `../outputs/roboproof-pedro-20260929-jdk` relative to the
repository root. That generated directory is deliberately outside both source
trees. The summary alone cannot replay the full corpus; regenerate it using
seed `1346716754`, count `2000`, the recorded source versions and a new output
directory, following the parent README.
