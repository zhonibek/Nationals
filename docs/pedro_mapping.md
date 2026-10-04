# Pedro reference mapping: geometry reuse is not Foresight equivalence

Audit date: 2026-09-28. Local HEAD: `f33c8392e163ac37814bb4e7cefdd5c229e89812`.

The reference directory inside `Nationals-work3` was deleted by the user. It was **not restored or modified**. Reference source was read from:

`C:/Users/kassi/.gemini/antigravity-ide/scratch/Nationals/PedroPathing-main/PedroPathing-main`

In this document, `P/` means that absolute reference root plus `core/src/main/java/com/pedropathing/`. All other paths are relative to `Nationals-work3`. These are mappings to the inspected local reference, not claims about a latest upstream release. Java execution and end-to-end Java/C++ parity are **UNVERIFIED**.

## Reference identity

Read-only `git hash-object --path=<tracked Pedro path> <surviving file>` comparisons matched `git rev-parse HEAD:<tracked Pedro path>` for these eight inspected files. Hashing did not write objects (`-w` was not used).

| Reference path | Matching Git blob |
| --- | --- |
| `P/algorithm/Foresight.java` | `54a8a3eaef657f39212e3d32bc61b1450409f292` |
| `P/algorithm/ForesightPowerAllocator.java` | `4b3ab32ef44a4db4a0b616373416f5d2eb4405c6` |
| `P/paths/curves/bezier/BezierCurve.java` | `21310bca060a76f4e9d766737d1ee6235e0d9bbe` |
| `P/paths/curves/Curve.java` | `6c3fdbd8caf0e74f6bafe79c11362cc6b3f54519` |
| `P/paths/PathTracker.java` | `27fc8c925a79fafe09a5dbf2d839313ccc29d105` |
| `P/follower/Follower.java` | `f93f888f30c111e04afee0ab0d2ad655c7cd9d21` |
| `P/localization/FusionLocalizer.java` | `5a652be547e98743488b7ae9852089b32caff26a` |
| `P/controllers/PIDController.java` | `9e6ca1a1e89d5a73b5bf68aae6a9edf9f53c3509` |

This identifies the compared files, not every file in the sibling tree. `P/paths/interpolator/Interpolator.java` and `P/math/Vector2D.java` were also inspected for heading/frame semantics but were not included in the eight-file identity check.

## Classification rules

- **SAME_SOURCE:** the local firmware and WASM compile the same C++ implementation. This never means Java Pedro source is shared.
- **FORMULA_EQUIVALENT:** the same mathematical primitive on a stated valid domain, after coordinate/unit conversion; not bitwise equality or identical edge-case/API behavior.
- **APPROXIMATION_DIFFERENT:** same intended quantity, but numerical method, sampling, precision or corner-case semantics differ.
- **CONCEPT_ONLY:** similar purpose or terminology, materially different control law/lifecycle.
- **NOT_EQUIVALENT:** source demonstrates different behavior/algorithm.
- **NOT_ACTIVE:** present locally but not selected by the current `main.cpp` call graph.
- **UNVERIFIED:** execution or equivalence evidence has not been produced. A source classification and this verification status can both apply.

## Which local implementation is actually active?

`src/main.cpp::runDiagnostic(7)` creates `pedro::BezierCurve` -> `nationals::BezierReference` -> `nationals::followReference` -> `nationals::Cascade` -> four wheel voltage loops. Autonomous routine index 2 invokes this diagnostic. Spline routines use the XDRIVE `LTVPathFollower` adapter into the same cascade.

`pedro::PedroFollower` is constructed globally and cancelled by `stopEverything`, but `main.cpp` never calls its `follow` method. Its implementation remains a separate callback-driven library follower. Therefore there are **two distinct local comparison targets**: active Bezier geometry/reference plus cascade, and inactive legacy PedroFollower. Neither is the reference Foresight algorithm.

## Algorithm mapping

| Feature | Reference evidence | Local evidence | Classification and evidence limit |
| --- | --- | --- | --- |
| Bezier position | `P/paths/curves/bezier/BezierCurve.java::generateBezierCurve/getDerivative` uses a cached Bezier basis/control-point matrix. | `src/subsystems/pedro/BezierCurve.cpp::getPoint`: cubic Bernstein fast path, linear fast path, general de Casteljau fallback. | **FORMULA_EQUIVALENT** for finite matching control points and t in [0,1], on their common supported domain. Different float/double arithmetic and API handling. Bounded local cubic formula probe PASS; Java comparison **UNVERIFIED**. |
| First/second derivatives | Same Java file, derivative order through `PolynomialMatrix`. | `BezierCurve.cpp::getDerivative/getSecondDerivative`: explicit cubic formulas and general derivative control polygons. | **FORMULA_EQUIVALENT** on the common valid domain. Local cubic value/first/second derivative probe PASS; general-degree and Java parity **UNVERIFIED**. |
| Tangent and normal | `P/paths/curves/Curve.java::tangent/leftNormal`: normalized derivative and (-y,x). | `BezierCurve.cpp::getTangent/getNormal`: same geometric construction. | **FORMULA_EQUIVALENT** when derivative is nonzero; zero-derivative normalization behavior and precision parity **UNVERIFIED**. |
| Signed curvature | Java `BezierCurve.java::curvature`: determinant divided by derivative magnitude cubed; zero if magnitude <1e-9. | Local `getCurvature`: same quotient, but zero if the cubed-magnitude denominator <1e-6. | **FORMULA_EQUIVALENT** away from guards; **NOT_EQUIVALENT** near small derivatives because thresholds differ substantially. Numeric Java/curvature probe **UNVERIFIED**. |
| Arc length / remaining distance | Java `approximateLength/subdivide`: adaptive midpoint subdivision, tolerance 1e-5, maximum depth 15, interpolated completion map. | Local `getLength` defaults to 40 uniform intervals; `getRemainingDistance` defaults to 25 over the remainder. Active `BezierReference` separately builds 512 intervals. | **APPROXIMATION_DIFFERENT**. Same geometric quantity, no established common error bound. Loop/cusp/adaptive-versus-uniform comparison **UNVERIFIED**. |
| Closest-point projection | Java `closestParameter`: samples 0,0.1,...,1 plus initial guess, then up to 10 Newton updates of squared-distance objective with a small denominator regularizer. | Local `BezierCurve.cpp::project`: 20-interval coarse scan then 18 golden-section iterations in each interval, keeping the best candidate. | **APPROXIMATION_DIFFERENT**; not a Newton port. Header still describes Newton-Raphson, but implementation does not. Active `BezierReference` does not use robot-position projection at all; it uses time/arc progress. Global-minimum guarantee and Java numeric parity **UNVERIFIED**. |
| Heading conventions | `P/math/Vector2D.java::rotate/toBodyFrame/toWorldFrame`; tangent heading from vector angle. | Public robot heading is clockwise from +Y; local path tangent/facing uses atan2(dx,dy). | **CONCEPT_ONLY** unless conversion is explicit. With the same field axes, h_local=pi/2-h_math and yaw rate changes sign. Test wrap and frame conversions before comparing controller outputs; runtime parity **UNVERIFIED**. |
| Constant/tangent/short linear heading | `P/paths/interpolator/Interpolator.java`: constant, tangent, and shortest-angle interpolation weighted by curve completion. | `include/subsystems/pedro/PedroPath.hpp`: matching conceptual modes; LINEAR uses total multi-segment sampled arc progress. Active `BezierReference` uses independent shortest-wrap heading with quintic distance/time progress. | **CONCEPT_ONLY** across complete APIs; constant/tangent primitives align after conversion, but completion sampling and multi-segment mapping differ. **UNVERIFIED** numerically. |
| Long linear and facing point | Java `Interpolator.longLinear` chooses the opposite of the shortest angular turn; `facingPoint` points from `curve.get(t)`. | Local `PedroPath::LONG_LINEAR` uses raw end-start, which need not be the long turn; FACING_POINT points from supplied robotPos. | **NOT_EQUIVALENT**, even before feedback control. Both modes are **NOT_ACTIVE** in the main Bezier demo. |
| Path chaining / modifiers | `P/paths/PathTracker.java` advances a deque and applies/reverts segment modifiers. | `PedroPath.hpp` stores curve segments and a path-wide heading mode; `PedroFollower.cpp` advances when projection and endpoint distance allow. Active main Bezier reference is one curve; `autoPath` chains generated spline samples. | **CONCEPT_ONLY**, not equivalent lifecycle or modifier support. End-to-end chaining parity **UNVERIFIED**. |
| Foresight prediction and braking | `P/algorithm/Foresight.java::getBrakeDisplacement/getCoastDisplacement/getVelocityToBrakeInTime/coast`: linear/quadratic directional braking model, projected pose, natural deceleration and optional coast/path-skip constraints. | Active `BezierReference` uses a precomputed quintic time law and curvature slowdown. Legacy `PedroFollower.cpp` uses min(maxVel,sqrt(2*maxAccel*remainingDistance)) in command space. | **NOT_EQUIVALENT** for both local targets. Similar stopping intent does not establish a calibrated Foresight prediction model or matching units. |
| Translational correction | `Foresight.java::translationalCorrection`: projected-pose displacement projected onto normal, body-axis configurable controllers, minimum-distance gate. | Active `Cascade.hpp`: Riccati pose-error feedback plus reference body velocity. Legacy `PedroFollower.cpp`: kP times full closest-point error plus velocity damping. | **NOT_EQUIVALENT**; active LQR is not Foresight/GVF. Legacy variable/comment names containing LQR do not demonstrate a Riccati-derived damping law. |
| Centripetal term | `Foresight.java::calculatePath` calculates curvature but passes `Vector2D.zero()` as normal feedforward into `allocatePowers`. | Legacy `PedroFollower.cpp` explicitly adds normal*targetSpeed^2*curvature*k_centripetal. Active `BezierReference` slows its profile using sampled curvature; active Cascade has no such explicit normal-force vector. | **NOT_EQUIVALENT**. Do not attribute an active v^2*kappa normal feedforward to this inspected Java Foresight call merely because the allocator accepts that argument. |
| Heading feedback | `Foresight.java` splits projected/current heading feedback and feedforward, with configurable controllers/static FF. | Active Cascade includes heading state in the Riccati controller. Legacy follower has gated/clamped integral, filtered error derivative, yaw damping and turn clamp. | **NOT_EQUIVALENT**. Timing and gain matching **UNVERIFIED**. |
| PID primitive | `P/controllers/PIDController.java` uses System.nanoTime, error integral and error derivative (or supplied velocity). | Active `Cascade.hpp::WheelVelocity` takes explicit dt, derivative-on-measurement filter, feedforward, conditional anti-windup and wheel/ramp limits. | **CONCEPT_ONLY** as feedback control, not an equivalent PID implementation. D is zero by default locally. |
| Allocation / saturation | `P/algorithm/ForesightPowerAllocator.java` prioritizes vectors based on deviations, splits heading contribution and uses drivetrain maxScaling and braking clamps. | Active `WheelVelocity` uniformly desaturates wheel targets then applies acceleration/voltage limits. Legacy follower combines vectors then uniformly scales final wheel peak. | **NOT_EQUIVALENT**; uniform normalization is not priority-based allocation. |
| Completion / holding | `Foresight.java::calculatePath` advances on its parametric condition; `P/follower/Follower.java::update` can transition to holdEnd. `calculateHold` evaluates heading/translation/velocity/timeout. | Active `followReference` waits for reference duration plus 150 ms pose/velocity dwell, then releases/stops. Legacy follower has final-segment remaining-distance/error/velocity dwell and callback brake. | **NOT_EQUIVALENT**. Releasing output is not a persistent hold controller. Completion timing parity **UNVERIFIED**. |
| Localization / delayed fusion | `P/localization/FusionLocalizer.java`: dead-reckoning localizer plus timestamped history, delayed measurement update and forward propagation. | Active `src/lemlib/chassis/odom.cpp`: four drive encoders and filtered velocities, no external measurements. C++ EKF/MCL are not selected in main. JS display EKF uses synthetic odometry. | **NOT_EQUIVALENT**; neither active path implements this delayed fusion contract. |
| Lifecycle / hardware abstraction | `P/follower/Follower.java`: pluggable Localizer, Drivetrain and Algorithm; update drives through selected state. | `main.cpp`, `HolonomicMotion.cpp`, `safety.cpp`: PROS callbacks/tasks, epochs, cancellation and leased output. | **CONCEPT_ONLY** at architectural level; different runtime/state/safety semantics. FTC hardware adapters were not executed; parity **UNVERIFIED**. |
| Local firmware vs browser controller | No Java counterpart implied. | `Cascade.hpp`, `BezierReference.hpp`, `QuinticSpline.cpp`, `BezierCurve.cpp` compiled by `tools/build-wasm.ps1` and exposed in `simulator/native/control.cpp`. | **SAME_SOURCE** for listed local algorithms; manifest checks PASS. Full firmware/browser equivalence **UNVERIFIED**, because odometry/safety/scheduling are not shared. |

## Evidence and limits

The audit verified the ten source hashes and WASM hash in `simulator/control-build.json`, and ran a read-only local cubic probe: two point sets times six t values, comparing value/first/second derivatives against direct Bernstein formulas with a 1e-4 tolerance. Details are in `simulator_audit.md`. The cubic fast path is itself Bernstein-form code, so this is a limited consistency check, not an independent Java oracle or proof across all degrees.

The existing `tests/control-core.test.js` includes a similar cubic check; its comment describes the C++ implementation as de Casteljau, although cubics use the fast path. That test does not instantiate the Java reference. This audit did not run it unchanged because it writes a report outside the three owned files.

**Explicitly UNVERIFIED:** Java execution; Java/C++ path, derivative, curvature, projection and heading fixtures; non-cubic/degenerate behavior; full control-output traces; FTC/VEX hardware equivalence; physical tuning and accuracy. No claim of a complete port is warranted by the `pedro` namespace or Bezier menu label.

## Recommended compatibility gates for the responsible owners

1. Decide whether the requirement is **Bezier geometry compatibility**, **Pedro-style behavior**, or a **Foresight algorithm port**. The current active implementation supports only a limited geometry/formula correspondence, not the last two as verified claims.
2. Preserve a read-only reference identity without restoring the user-deleted worktree tree. Record exact source revision/blob identities in any external fixture provenance.
3. Produce coordinate-normalized Java/C++ fixtures for ordinary curves, repeated control points, zero derivatives, loops, projection ties, heading wrap, long-linear and facing-point semantics. Define numerical tolerances and approximation error before declaring equivalence.
4. For any proposed Foresight port, separately test projected braking/coasting, correction, heading split, priority allocation, segment modifiers and path-to-hold transition. Matching curve points is not sufficient.
5. Keep controller comparison separate from sensor/plant comparison. Use fixed explicit input histories and document any reference controllers' wall-clock dependence. Physical acceptance requires independently measured robot truth.

These are handoff requirements, not changes made or tests claimed complete by this documentation-only audit.
