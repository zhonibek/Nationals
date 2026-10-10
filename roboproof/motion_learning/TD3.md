# Experimental deterministic reference learner

This is an explicitly separate `td3-beta-mean-reference-v1` experiment. PPO and the production controller remain available and unchanged by default. It does not qualify a policy for deployment or claim AMD, game-score, obstacle/camera or physical-robot performance.

The motivation is retained training-world evidence: the exact pre-update PPO actors succeeded on19/27 recorded stochastic rollouts but16/27 deterministic-mean rollouts. Python and JavaScript mean execution agreed on every reason. This small cohort supports investigating objective alignment, not a proven explanation for held-out failures.

## Architecture

- Reuse the original34-value sensor/reference contract and four bounded reference actions, with the existing four-frame145-feature encoder and two32-neuron actor layers by default.
- The eight actor outputs parameterize a Beta mean. TD3 optimizes that deterministic mean directly through its first learned Q function; it does not use PPO log-probability ratios or label itself PPO.
- Two independent32x32 Q networks estimate returns from encoded public observations and actual applied actions. They are value estimators, not a physics model or a replacement simulator.
- Real transitions, terminal outcomes and rewards always come from the original Simulator through the existing persistent iraLIB C++ LTV-LQR/wheel PI/feedforward cascade.
- Training exploration adds independent Gaussian noise on every action axis, clipped only to the existing action contract. Deployment remains the unprojected bounded actor mean; no action head is frozen.
- Bellman targets use the smaller target-Q estimate, bounded target-action smoothing and intrinsic task terminals. Actor/target updates occur every second critic step, with Polyak factor0.005. This follows the core ideas in [the original TD3 paper](https://arxiv.org/abs/1802.09477) and [the primary algorithm equations](https://spinningup.openai.com/en/latest/algorithms/td3.html), with a Beta-mean actor and project-specific training priors.

## Bounded data and reproducibility

The default budget is64 complete collection updates,256 minimum accepted rows per collection,50000 actual outer decisions and900 seconds shared across at most three seeds. Eight initial collections use the existing measured three-pace controller teacher. Every teacher attempt, complete trajectory, partial collection and discarded query is charged and exposure-journaled; teacher outcomes are privileged training labels only.

The default replay ring holds1024 encoded transitions, sampled in64-row batches. Each TD3 collection performs32 twin-Q gradient steps and16 deterministic-actor steps. The training-only public-deadline/pose-budget anchor remains explicit, with all four actions learnable. Historical PPO returns and failed evaluation artifacts are never overwritten or relabeled.

Checkpoints bind source/engine/runtime/configuration, actor, target actor, both critics/targets, optimizer moments, replay/cursor, measured anchor, Python/Torch RNGs, complete history and actual/discarded work. Normal resource interruption rolls back uncommitted RNG/data/learning state, retains spent cost/exposure, and can resume the same declared method exactly. Source/scale/configuration drift fails closed. Checkpoints are limited to4MiB and exported actors to the existing bounded reader limits.

## Local command

### Optional experience retention

`--replay-strategy uniform-reservoir-v1` retains a fixed-capacity uniform sample over all accepted learning rows instead of only the recent ring. At rowN draw an integer0..N-1; replace that slot only when below capacity. Every accepted row increments seenRows, whether retained or not. No reward/outcome selection or new environment query is introduced. Replay schema3 binds method/count/cursor; the existing saved Torch RNG provides exact replacement continuation. Old ring/schema2 stay default and sampler changes fail source/config resumption checks. Reservoir retains older behavior, so the declared multi-step off-policy approximation still applies.

Four workspace-only reservoir kernel tests and14 cross-component kernels pass. A separate original-engine standard CLI full4 versus prefix2/resume4 probe also confirms matching actors, both critics/targets/optimizers/replay/RNG/history/work; all outputs stay under the workspace and the unchanged child/stdin/stdout bridge needs no additional privilege. The denied elevated tempfile-based unittest and full regression remain unexecuted, not relabeled passed. Bounded historical-stream rehearsal shows context diversity, not stronger movement. Current-source fixed64 ring reproduction and all pilot/fresh-seed/final gates still apply. This mode is experimental, opt-in and unqualified for policy promotion.

The fixed64 reservoir pilot subsequently passed both complete development suites:28/32 versus controller27/initial25 and43/64 versus controller/initial42, zero success/contact regressions,11.1401%/7.9431% all-world effort-proxy improvement. Ring actor reproduced exactly, initial actors/units match and every alpha/beta row adapts. Fresh seeds17001/17002/17003 are predeclared after complete bounded manifest inventory; every seed/full gate and a new untouched original128-case final remain required. A single-seed pass cannot promote this mode or replace pending current-source integration/full-suite validation.

The fresh reservoir replication completed64/64/64 but reached28/28/27 on original32 and42/43/42 on mixed64. Success/contact regressions are zero and all effort gains meet5%, but several seeds tie controller success. The every-seed gate therefore fails; no final corpus is prepared/consumed or actor promoted. More varied replay and favorable efficiency alone do not establish better movement. The standard-TD3 exposure directory is now included in future exclusion audits; old source identities/results remain historical and immutable.

### Optional critic outcome sampling

`--critic-terminal-fraction 0.25` mixes25% terminal-horizon sampling with75% uniform replay, only for the critics. Each row has probability `q = (1-f)/N + f/T` when its recorded horizon ends intrinsically, otherwise `(1-f)/N`; if there are no terminal rows, use uniform replay. Inverse weights `1/(N*q)` keep the expected critic loss and its unclipped gradient equal to the uniform-row objective. Do not normalize weights within a minibatch, clip them, or interpret shared terminal horizons as independent episodes. The mixture is bounded0–0.5 and weights stay at most2. Default0 retains the original sampler/RNG/update numerics.

The actor and its public priors receive a separately drawn uniform replay batch, never the terminal-biased critic batch. All recorded observations/actions/rewards remain from the original Simulator/iraLIB, and terminal selection provides no additional actor input or environment query. Sampling definition/fraction are configuration/source-bound; the existing replay/target/optimizer/RNG checkpoint provides exact resumption. Four focused formula/default-RNG/critic-weight/actor-separation/real-Simulator resume tests pass. This hypothesis addresses terminal-value fitting, not proven independent improvement; full regression and fixed-candidate development/fresh-seed/final gates remain mandatory.

```powershell
& roboproof/gpu/.venv/Scripts/python.exe -m roboproof.motion_learning.td3_train `
  --directory roboproof/runs/motion/research/td3-pilot `
  --seed 201 --seed-count 1 --updates 64 --max-steps 50000 `
  --rollout-steps 256 --max-seconds 900 --warm-start-updates 8 `
  --feature-transform deadline-context-v1 --curriculum mixed-full-reach-v5 `
  --objective finite-total-return-settle-margin-v4 `
  --gradient-steps 32 --replay-capacity 1024 --exploration-std 0.05
```

Use a new explicitly named directory and a predeclared protocol. `--resume <run-id>` reuses the same source/configuration and saved episode boundary, never starts over because an observation wait expired. A resource stop is not a completed selected checkpoint.

## Mandatory improvement gates

The initial fixed64 TD3 pilot completed all real original-engine learning but reached27/32 versus controller27 and43/64 versus42, with zero success/contact regressions and11.4185%/8.2256% effort gains. The higher-success gate fails on original32, so no replication/final/promotion follows; paired PPO control reproduced exactly. A full replay/teacher audit found weak Q gradients relative to priors and discrepancies from exploratory behavior returns, not current deterministic-Q ground truth or proven failure causation.

Optional `--n-steps 8` aggregates measured complete-episode rewards before bootstrapping. Default1 preserves the prior one-step rule. Replay schema2 stores each horizon1–32, original start action/sensor state and matching end state; success/fault/intrinsic deadlines stop horizons and resets are never crossed. Later exploratory behavior actions introduce an explicit off-policy approximation. This is a multi-step TD3 variant, not a new simulator or canonical one-step equivalence. It adds no physical query or actor truth. The predeclared one/eight-step comparison must reproduce the one-step actor and retain every unchanged improvement gate.

The fixed64 one/eight-step study completed with identical initial actors and all eight alpha/beta rows learning after imitation. One-step byte-reproduced its retained actor. Both variants reached27/32 and43/64, with zero success/contact regressions; eight-step effort-proxy reductions were11.4504%/8.2272%. Neither improves original32 success over controller27, so both fail the full development gate. Longer credit propagation alone did not establish better movement. No fresh replication or independent final is authorized by these results.

A subsequent full-buffer read-only diagnosis measured eight-step actor gradient norms0.01183 for Q versus0.09251 for the deadline prior and0.04777 for the teacher anchor. These are hypotheses about optimization balance, not sampled-update or causal evidence. A predeclared same-source fixed64 pair tests anchor10->1 and deadline40->4 together, leaving residual KL0.02 and every other method/interface/resource/gate unchanged. Only the declared balanced candidate may advance, and the control must byte-reproduce the previous eight-step actor. No inference projection, gradient-norm-based adaptive weight or action head removal is introduced.

The lower-weight candidate subsequently passed both single-seed development gates:28/32 versus controller27/initial25 and43/64 versus controller/initial42, zero success/contact regressions,9.5686%/6.8539% effort-proxy reduction. Current control reproduced exactly, initial actors/units match and every alpha/beta row learned. This is only a pilot; the public verification flag correctly stays false for one training seed. Three predeclared fresh seeds16001/16002/16003 are required next, then a new untouched original128-case final. No policy is promoted from this pilot.

Fresh balanced seeds16001/16002/16003 completed64/64/64 with every alpha/beta row adapting. Original32 reaches28/28/28 with zero regressions, but seed16002 effort improvement4.8547% and time improvement0.7006% both miss5%. Mixed64 reaches43/43/42; seed16003 ties controller/initial42. Thus the complete three-seed replication fails: no final corpus is prepared/consumed and no policy is promoted. Favorable seeds or rounding cannot repair this gate failure. Full-replay value/gradient diagnoses remain exploratory, not current deterministic-return truth or a demonstrated remedy.

First compare a current-source PPO control and the fixed TD3 candidate on every original32 and mixed64 development case, preserving the complete failures and all independent comparators. Reproduce the retained PPO control actor before judging the alternate algorithm. Then require every predeclared fresh training seed on both complete development suites, followed by a new untouched exposure-bound original128-case final with the original family mix and frozen candidates.

The critic-terminal-mixture0.25 pilot also failed original32 success:27/32 versus controller27, despite43/64 versus42, zero regressions and11.8445%/8.9543% effort-proxy gains. Uniform control reproduced exactly and remains the previously failed-replication method, not a newly qualified replacement. Common retained terminal-row fitting improved modestly, not movement success. No replication/final/promotion follows. A separately predeclared all-terminal-group value-regression diagnosis investigates fitting versus cross-group stability, without modifying actors or physics; it cannot qualify a policy.

Each seed must improve success over the existing controller and its initial untrained actor, have zero controller-success/contact regressions, and improve all-world mean time or electrical effort proxy by at least5%. The electrical proxy is not calibrated energy. No best seed/checkpoint, weaker tolerance, easier subset, consumed-final reuse or favorable metric substitution is allowed. Gradient tests, low Q loss, training-world success and this implementation alone do not promote a policy. Bounded tactical strengthening follows genuine movement proof; product UX/voice/avatar remain planned-only.
