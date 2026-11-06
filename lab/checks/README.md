# lab/checks: deterministic graders

Planned. Each check reads measured results from the simulator, the rules engine or a bench log and returns pass or fail with a reason.

- **Examples:** final pose error below a limit; overshoot below a limit; score equals the rules-engine score; the learned and scripted tactics compared on the same held-out starts.
- Checks never call a language model. A model may explain a result to the student, but it can't change it.
- Each check result is stored with the lesson ID, the measured values and a hash of the inputs, so a teacher can replay it.
- Tests: `tests/lab-checks.test.js`.
