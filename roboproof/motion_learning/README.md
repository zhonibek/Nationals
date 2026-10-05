# Canonical motion learner

This is a real, bounded CPU reinforcement learner inside Nationals-work3, not another physics model. Python's PyTorch actor-critic receives the existing sensor-only 34-value observation and sends four bounded reference actions through `MotionEnvironment` to the original Simulator and source-checked iraLIB C++ controller. The JavaScript plant/controller runs at 100 Hz; decisions normally run at 20 Hz. The separate historical failure predictor is not used.

## Run explicitly

From the repository root, reuse the existing pinned `torch==2.8.0` CPU environment:

```powershell
node roboproof/cli.js motion-check
& 'roboproof/gpu/.venv/Scripts/python.exe' -m roboproof.motion_learning.train --seed 42 --seed-count 3 --updates 8 --max-steps 4096 --max-seconds 180
node roboproof/cli.js motion-compare-learned
node roboproof/server.js 8766
```

Open the existing dashboard's **Motion lab**, click **Refresh training history**, then inspect the controller/untrained/learned comparison. Select **Learned PPO · explicit experiment** to run a new task and open its recorded actions in the original 2D/3D Simulator. The default is still controller-only. A successful individual experiment does not certify learned improvement.

The default budget is three independent seeds, at most eight updates and 4,096 policy steps per seed, with a shared 180-second collection deadline. Environment access has its own timeout; one already-started bounded PPO update/checkpoint/teardown may finish after the collection deadline. There is one bridge worker and one CPU training thread. No cloud call, firmware command, AMD provisioning or model download occurs. The browser never launches a Python process, installs dependencies, or starts training on load. A CLI run is explicit and bounded; refreshing reads disk only.

## Algorithm and boundaries

- Two shared 32-neuron tanh hidden layers, eight Beta concentration outputs (four alpha/beta pairs), and a scalar value head. Deterministic evaluation uses the Beta mean mapped to `[-1, 1]`; stochastic rollouts use the distribution itself. This is a small deep RL model, not an LLM.
- PPO clipped likelihood ratio (0.2), four optimization epochs with 64-row minibatches, Adam at 0.0003, gradient norm cap 0.5, latent-distribution entropy coefficient 0.001, gamma 0.99 and GAE lambda 0.95. Observations use declared fixed engineering-unit scales and clip to ±10; no normalization is fitted to frozen evaluation data.
- History's `maximumGradientNorm` is the pre-clipping norm returned by PyTorch, not the applied post-clipping norm or a claim that the cap was exceeded.
- Actual simulator reward is accumulated at policy cadence. Value bootstrap is zero for genuine terminal states, but uses the final observation for time-limit truncations; advantages never flow into the next reset episode. A shortened final cadence uses the same policy-step discount; no exact continuous-time discount claim.
- Curriculum: short near-forward reach, varied starting heading/direction/goal heading, then supported mass/traction/slip/battery-resistance variation. The frozen v1/v2 seeds and tasks are excluded. Evaluation does not select hyperparameters or checkpoints. Initial-policy evidence is retained for each seed.
- No obstacle observation, camera input, independent slip sensor, pickup action or game reward is present in this contract. This is reference adaptation around LTV-LQR and wheel PI/feedforward, **not general obstacle navigation, learned motor control, or learned VEX scoring**. Sensor ambiguity and limited residual bounds can prevent improvement; a small smoke run is not a trained product.

Algorithm references: [original PPO paper](https://arxiv.org/abs/1707.06347), [PyTorch 2.8 distributions](https://docs.pytorch.org/docs/2.8/distributions.html), [PyTorch reproducibility limitations](https://docs.pytorch.org/docs/2.8/notes/randomness.html). No third-party RL implementation was installed or copied; the existing PyTorch dependency is reused.

## Saved progress and resume

Ignored local outputs live under `roboproof/runs/motion/learning/<run UUID>/`. Each seed retains its initial JSON actor, immutable numbered actor exports, atomically replaced latest PyTorch checkpoint and summary. The run manifest and local latest pointer are updated at committed checkpoints. The checkpoint includes actor/critic, Adam state, CPU Torch RNG, world-generator RNG, configuration, episode/step/update counters, history, training-source hashes and exact engine/WASM identity. JSON actor exports permit bounded JavaScript inference without Python or pickle on the web server.

Resume at a committed complete-rollout/episode boundary:

```powershell
& 'roboproof/gpu/.venv/Scripts/python.exe' -m roboproof.motion_learning.train --resume RUN-UUID --seed 42 --seed-count 3 --updates 8 --max-steps 4096 --max-seconds 180
```

The seed/curriculum/rollout settings, source hashes, Torch version and canonical engine identity must match. Update/step budgets may increase explicitly but never decrease. Partial uncommitted rollouts are discarded and RNG state rolls back to the saved boundary. No arbitrary mid-episode WASM restoration is promised. Normal exceptions/interruptions close the owned bridge; force-killing the entire Python process may leave its child until manually cleaned up, so prefer Ctrl+C.

Model files, steps, workers, checkpoint count and read sizes are bounded. These are local trusted artifacts, not a public arbitrary-checkpoint upload endpoint. Changed simulator sources intentionally invalidate old training/replay identities instead of silently pretending old artifacts match a new plant.

## Improvement is a separate gate

`motion-compare-learned` freezes all committed policies and compares the existing controller, each seed's initial untrained actor, and each learned actor on the same 24 frozen worlds. It preserves every failure, position/heading error, elapsed time, contact duration and electrical effort proxy; the proxy is not calibrated energy consumption.

The existing predeclared gate requires higher success than both controller-only and untrained policy, zero baseline-success regressions, no per-world contact increase, and at least 5% all-world mean time or effort improvement. Three independent training seeds must each pass. A gradient update, lower training loss, a single favorable seed or higher training reward does not unlock **Use verified policy**. The first seed is the predetermined experimental policy; there is no best-seed selection using the frozen corpus. The experiment option remains explicit even when the improvement gate fails.

After inspecting this frozen benchmark, never tune against its individual cases and reuse it as fresh evidence. Further development should use training/development-only worlds and reserve an additional untouched final corpus before claiming new held-out improvement. Physical calibration, game mechanics, genuine AMD training, perception and hardware safety are separate future gates.

## Tests

```powershell
& 'roboproof/gpu/.venv/Scripts/python.exe' tests/roboproof_motion_learning_test.py -v
& 'roboproof/gpu/.venv/Scripts/python.exe' tests/roboproof_motion_env_test.py -v
node --test tests/motion-learner.test.js tests/motion-policy.test.js tests/motion-preparation.test.js tests/motion-api.test.js
```

Tests exercise actual simulator gradients, byte-identical interrupted/resumed exported policies, Python/JavaScript deterministic action parity, terminal/truncation handling, budget exhaustion, source mismatch, malformed artifacts, replay, provenance and loopback API protection.
