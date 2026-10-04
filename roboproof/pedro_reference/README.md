# Bounded Pedro differential check

This executes **surviving, unmodified Pedro Java source** against the repository's
**compiled C++ `simulator/control.wasm`**, through `control_curve_eval`. It is not a
JavaScript oracle and does not restore/copy the deleted Pedro source tree. It
checks only cubic Bezier position, first/second parameter derivatives, and signed
curvature. It does not establish follower, trajectory, projection, length, or
whole-robot equivalence.

## Run

From the repository root, with Node on PATH (PowerShell example):

```powershell
node roboproof/pedro_reference/run.js `
  --reference 'C:\Users\kassi\.gemini\antigravity-ide\scratch\Nationals\PedroPathing-main\PedroPathing-main' `
  --out 'C:\Users\kassi\.gemini\antigravity-ide\scratch\pedro-generated-run-001' `
  --seed 1346716754 --count 2000
```

`--out` is required, must be a **new directory outside both source trees**, and
is never overwritten. Symlinks/junctions are resolved before the containment
check. All generated artifacts, including `.class` files, stay there. No source
files are patched or generated during execution. Reference and output paths may
contain spaces. No Gradle, downloads, Android SDK, or third-party dependencies
are used.

Exit codes: **0 PASS**, **1 FAIL** (numeric discrepancies), **2 BLOCKED** (missing
Java/compiler/reference/WASM, stale build, failed compilation/execution/protocol,
or invalid options). BLOCKED has zero completed comparisons and never claims
Java equivalence. Invalid options/output paths are reported on stderr before a
report directory can be created. A failed run must not be treated as a pass just
because no discrepancies are available.

## Java and minimal dependencies

Discovery checks `JAVA_HOME`, `JDK_HOME`, PATH (including resolved Java shims),
common Program Files JDK vendors, Android Studio's `jbr`, and `~/.jdks` on Windows;
standard JVM directories on Linux/macOS. It inspects at most 16 runtime candidates.
For an unusual installation, use `--java '.../bin/java.exe'`, optionally with
`--javac '.../bin/javac.exe'`. Explicit `--java` disables fallback discovery.
`--javac` requires the corresponding `--java`.

With an adjacent `javac`, the runner compiles the allowlisted sources and
`PedroProbe.java`, then invokes that class in the matching JVM. Java 8 JDKs are
sufficient for the current sources. Without `javac`, Java 11+ source-file mode
launches `CompileAndRun.java`, which invokes the JDK's **actual JavaCompiler** and
loads the resulting classes in an isolated loader. A Java 8 **JRE** cannot compile
these sources and is BLOCKED. A newer runtime without compiler capability is
also BLOCKED; it is never replaced by a port, stub or precomputed answer.

The dependency closure is 15 source files under `core/src/main/java/com/pedropathing`:

- `paths/curves/bezier`: `BezierCurve`, `BasisMatrixSupplier`, `PolynomialMatrix`.
- `paths`: `TValue`; `paths/curves`: `Curve`.
- `math`: `Matrix`, `Vector`, `Vector2D`, `Pose`, `Twist`, `Velocity`.
- `utils`: `Angle`, `Pair`, `Utils`, `BijectiveMap`.

`Pose` brings in `Twist`, `Velocity`, and `Angle`, even though the probe uses the
Vector2D constructor. The full classes are compiled, not extracted methods.
The bridge calls `get(t)`, `derivative(t)`, `getDerivative(2,t)`, and `curvature(t)`;
it implements no Bezier formulas. `-proc:none` and an isolated class/source path
prevent annotation processors or implicit source dependencies. Reference hashes,
adapter/launcher/runner hashes, Java command/version/diagnostics, and validated
WASM/source hashes are recorded for provenance. Compilation errors from future
Pedro API/dependency changes are BLOCKED, not silently adapted.

## Domain, bounds and tolerances

Mulberry32 with an explicit unsigned 32-bit seed generates 1–10,000 random cubic
cases (default 2,000), plus 77 fixed cases. Fixed cases cover endpoints,
near-endpoints, mirrored CW/CCW turns, loops, inflection, constant curves,
stationary endpoints, a cusp, tiny speed, and a large coordinate offset. Random
control points are bounded to ±144 in a common arbitrary Cartesian unit, with
scales 0.01, 1, and 144; `t` is in [0,1]. Inputs are quantized once to float32 and
the **same numeric values** are fed to both implementations. This isolates
Java-double versus C++-float arithmetic rather than input-rounding differences.
Out-of-range/nonfinite inputs and other curve degrees are outside this scope.

Each component passes if `abs(wasm-java) <= absolute + relative * max(abs(wasm), abs(java))`:

| Output | Absolute tolerance | Relative tolerance |
| --- | ---: | ---: |
| x, y | 5e-5 | 3e-6 |
| dx/dt, dy/dt | 2e-4 | 4e-6 |
| d²x/dt², d²y/dt² | 5e-4 | 4e-6 |
| Signed curvature | 5e-5 | 5e-4 |

The absolute terms accommodate float cancellation near zero at the bounded
coordinate scales; relative terms allow accumulated float arithmetic error.
Curvature is more sensitive because it divides by speed cubed. Tolerances are
fixed, not fitted to results or enlarged after failures. Nonfinite outputs always
fail, including matching NaNs or infinities. Near-singular cases outside the
explicit cutoff policy may legitimately expose discrepancies; these are retained,
not discarded or retried with a different seed.

### Intentional conventions

Both inspected primitives use signed `cross(first, second) / |first|³`, positive
for counterclockwise Cartesian curvature. **No absolute-value comparison or
blanket sign negation is permitted.** Clockwise heading measured from +Y, used
by the robot control layer, has the opposite angular-rate sign; that intentional
heading convention is not applied to primitive Cartesian outputs. Mirrored
fixtures and a sign-mutation test guard against accidentally hiding a sign bug.

C++ intentionally returns zero when `|first|³ < 1e-6`, while Java returns zero
only when `|first| < 1e-9`. When both measured speeds are below 0.00999 (a guard
band below 0.01), and Java speed is at least 1e-9, the expected C++ curvature is
**exactly zero**. Every nonzero-Java/zero-C++ occurrence is logged separately as
an intentional cutoff difference, with complete input and raw results. Raw
curvature error remains in statistics; adjusted error uses zero. Other fields
are still checked. Cases in the narrow cutoff boundary band are compared
normally, not excused. This policy is not a claim of equal low-speed curvature.

Java has a 256 MiB heap limit; each probe has a 3-second timeout and each compile
or run a 30-second timeout, within a 120-second overall Java budget. Captured
subprocess output is capped at 16 MiB. The Java bridge also enforces a 10,100-case
limit; the harness generates at most 10,077. These are bounded checks, not a proof.

## Artifacts and tests

- `inputs.json`: every exact case, ID, label, points and t; seed/count are in the report.
- `wasm-results.json`, `java-results.txt`: actual raw backend results (when available).
- `report.json`: status, provenance, per-component count, nonfinite/failure count,
  maximum raw/adjusted absolute error, RMS error, worst tolerance ratio/case ID,
  every discrepancy with full input/both outputs/failed fields, and intentional differences.
- `classes/`: only generated Java compiler output. BLOCKED runs retain whatever
  evidence was available, including Java attempts and their diagnostics.

Reproduce a corpus with the recorded seed/count and the same recorded source and
WASM hashes, using a new output directory. Each discrepancy also contains its
complete standalone primitive input.

```powershell
node --test tests/roboproof-pedro.test.js
```

Tests cover corpus bounds/determinism, strict Java protocol, tolerance/sign/nonfinite
failures, cutoff accounting, actual WASM outputs, stale-build detection, output
safety, missing-Java CLI status and a real 2,077-case integration check. Integration
uses `PEDRO_REFERENCE` (default: the surviving sibling checkout above), optionally
`PEDRO_JAVA`/`PEDRO_JAVAC`. If prerequisites are absent, it explicitly reports
**BLOCKED as a skipped integration test**, not a Java/WASM PASS. Unit-test success
alone is not evidence that the Java differential comparison ran; use the CLI's
status/exit code as the strict automation gate.
