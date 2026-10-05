# Override elements: geometry and fidelity

Reviewed 2026-10-04 for V5RC Override 2026–2027, manual v2.0. Sources below are links, not bundled vendor assets. The online manual is complementary; the official PDF and official Q&A govern competition rulings.

## Official sources

- [Competition resource page](https://www.vexrobotics.com/v5/competition)
- [Field CAD ZIP](https://link.vex.com/docs/26-27/v5rc/field-cad)
- [Official manual PDF](https://link.vex.com/docs/26-27/v5rc/game-manual)
- [Online manual and Appendix A drawings](https://www.vexrobotics.com/override-manual)
- [Pin specification drawing](https://content.vexrobotics.com/docs/2026-2027/override/online-manual/assets/image/PinSpecs.png)
- [Cup specification drawing](https://content.vexrobotics.com/docs/2026-2027/override/online-manual/assets/image/CupSpecs.png)
- [Scoring Element Kit 276-9255](https://www.vexrobotics.com/276-9255.html)

The official CAD link was found on the VEX competition page. The direct download returned HTTP 403 and the browser download did not complete. No CAD archive, converted mesh or original drawings were imported, redistributed, or silently substituted with community files. Current meshes are independently generated procedural approximations based on factual dimensions; they are not vendor CAD.

## Drawing-based dimensions (inches)

| Feature | Drawing | Current use |
| --- | --- | --- |
| Pin overall height | 6.50 | 6.50 |
| Pin maximum central flange diameter | 3.16 | radius 1.58 for visual flange and conservative contacts |
| Pin end diameter | 1.40 | radius 0.70 in profile |
| Pin shoulder diameter | 2.35 | radius 1.175 in profile |
| Pin end collar length | 0.64 | procedural profile transition at 2.61 from center |
| Cup overall height | 6.48 | 6.48 for mesh; shared stack/contact height remains approximately 6.50 |
| Cup rim diameter | 3.16 | radius 1.58 |
| Cup waist diameter | 2.32 | radius 1.16 |

The glossary's nominal 1.6-inch Pin diameter does not describe the wider central flange visible in the engineering drawing. The previous 0.8-inch circular Pin proxy missed this flange and the ends of lying pieces. Contact proxies now use conservative sampled capsules oriented with tilt/yaw; they are **not exact hollow mesh collision surfaces**.

The Cup shell thickness (0.06 inches), fine profile transitions, material appearance, and default Pin/Cup masses (0.073/0.078 kg) are modeling assumptions, not measured or certified vendor specifications. The product listing provides a combined kit weight, not individual calibrated masses. Restitution, friction, support/toppling and manipulator geometry remain approximate. Procedural Pin flanges are rotational approximations; small ribs, hollow Pin internals and manufacturing details are not exact CAD.

## Scoring and interaction

The existing rule engine is authoritative for the on-screen calculator: visible red/blue halves contribute 5, owned yellow halves 10, midfield robots 8, and autonomous bonus 0/6/12. A Cup provides support/occlusion but no independent points. Yellow ownership is determined by toggles or midfield robot counts; the opaque Cup half can conceal a Pin half. Appearance controls do not change these rules.

Final score and its breakdown freeze together after rest or the five-second grace deadline. A later confirmed autonomous ruling updates the frozen autonomous bonus without re-evaluating physical objects. Referee intent/violations are not automatically converted to negative points.

Release and scoring are distinct: placement releases the held object with robot velocity, then gravity/alignment/stack compatibility determine whether it nests. Pickup is only possible near the front fork at a compatible height. Prepared tutorial scenes are explicitly separate from official field starting positions.

## Remaining fidelity gates

1. Import legally usable official STEP/CAD, convert and verify each mesh's scale, axis and pivot against the drawings.
2. Replace conservative contact proxies with measured hollow/convex decomposition while preserving deterministic fixed stepping and browser/headless parity.
3. Calibrate individual masses, centers of mass, foam friction, restitution and robot/manipulator contact dimensions.
4. Model collision of held objects, finite grip strength, distributed rigid-body torque, de-scoring and collapse of placed stacks. Current placed stacks remain rule-state constraints, not fully dynamic rigid bodies.
5. Independently verify full field arrangement, loader geometry, partial nesting/SC2 edge cases, continuous rotation and all referee-dependent rules against vendor references and recorded real experiments.

Do not pool old motion traces or game results with the new source identity. Historical saved replays are retained, but the provenance checker correctly rejects stale source hashes; rerun experiments/readiness under the current engine.
