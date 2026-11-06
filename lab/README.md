# AI Engineering Lab: MVP outline

**Status: proposal only.** This folder sets out the structure of the kids' learning app for [Microsoft Imagine Cup 2027](../docs/MICROSOFT_IMAGINE_CUP_PLAN.md). It has no app code yet, and nothing in it is served or tested. The product and business plan are in [AI_ENGINEERING_LAB_STARTUP_PLAN.md](../docs/AI_ENGINEERING_LAB_STARTUP_PLAN.md).

The lab wraps the existing simulator in short lessons. A student gives an AI a task, the rules engine checks each decision, the browser runs the robot's own C++ controller, and the student decides from measured results. The last lesson moves the routine to a real VEX V5 robot.

## Principles

1. **Measurement decides.** Lessons are passed by deterministic checks on simulator or bench measurements, never by a model's opinion.
2. **The AI fails closed.** Models only choose from allowlisted tool calls. An invalid answer stops the run, and the student sees why.
3. **Reuse, don't fork.** Lessons drive the existing `simulator/` and `roboproof/` modules. The physics, the controller and the rules engine are not copied.
4. **Child safety first.** Pseudonymous student IDs, no open chat, Content Safety on every model call, and a teacher approves real-robot runs.
5. **Azure in production, local for development.** Training jobs run only on approved remote GPUs (Azure Machine Learning), never on a developer laptop (see [`AGENTS.md`](../AGENTS.md)).

## Folder map

```text
lab/
├── README.md        this outline
├── app/             student and teacher web app shell (static; embeds the simulator)
├── lessons/         lesson content as data: text, task, starting state, checks
├── checks/          deterministic graders over measured results
├── tutor/           AI tutor and coordinator gateway: prompts, tool allowlist, providers
├── classroom/       classes, pseudonymous students, progress, teacher dashboard data
├── bench/           VEX V5 test-bench protocol and the simulation-to-reality comparison
└── azure/           infrastructure plan (Bicep later), service map, cost tracking
```

Tests for the lab belong in the repository's `tests/` folder as `tests/lab-*.test.js`, because `tools/run-tests.js` only runs `tests/*.test.js`.

## What it reuses

| Lab part | Existing code | Change needed later |
| --- | --- | --- |
| Lesson sandbox | `simulator/index.html`, `simulator/simulator.js`, `simulator/engine.js`, `simulator/control.wasm`, `simulator/control-runtime.js` | An embed mode: hide expert panels, expose lesson hooks (start state, events, measurements) |
| Motion lessons (1–3) | `simulator/motion.js`, quintic and Bézier paths, Pedro Pathing C++ rewrite | Lesson presets and readouts for tracking error and overshoot |
| Rules lessons (4) | `simulator/override.js` (`OverrideGame` interact and score) | Small scenario loader for lesson start states |
| Learning lesson (5) | `roboproof/tactical_task.js`, `roboproof/tactical_policy.js`, `roboproof/tactical_environment.js`, `roboproof/models/tactical-ppo-seed736001` | A held-out comparison view; bounded retraining jobs on Azure Machine Learning |
| Coordinator lesson (6) and tutor | `roboproof/tactical-coordinator.js`, `roboproof/nemotron.js` | A Microsoft Foundry provider. `roboproof/nemotron.js` accepts only Nebius or loopback hosts today. |
| Evidence lesson (7) | RoboProof run records and hashes (`roboproof/core.js` and related modules) | A student-readable evidence card |
| Bench lesson (8) | PROS firmware in `src/`, `include/` | A bench run sheet and log upload through the bridge PC |
| Server | `tools/start.js`, `roboproof/server.js` | Serve `/lab/` and add `/api/lab/*` routes; containerize for Azure Container Apps |

## MVP v0 scope (target: December 20, 2026)

- Lessons 1–3 (drive and measure, control loops, paths), each with checks.
- A teacher view with class progress (pseudonymous).
- It runs locally with `npm start` and is deployed to Azure: Static Web Apps, Container Apps, Container Registry, PostgreSQL, Blob Storage, Key Vault and Application Insights.

**Next, before the pilot (target: January 11, 2027):** lessons 4–6, the tutor behind Content Safety, consent and privacy notice, and Kazakh and Russian lesson text reviewed by a teacher.

**Not in the MVP:** open chat, automatic robot control by the AI, camera perception, Cosmos video reasoning, and payments.

## Suggested order of small changes

1. Lesson format and one example lesson as data (`lessons/`), with a schema test in `tests/`.
2. Simulator embed mode and lesson hooks.
3. Check runner for lessons 1–3 (`checks/`).
4. Static app shell and the `/lab/` route.
5. Classroom data model and teacher view.
6. Foundry provider for the coordinator and tutor, compared with the existing Nebius results using the same seeds.
7. Azure infrastructure files and the first deployment.
