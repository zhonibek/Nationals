# Episode-balanced movement study — 2026-10-16

## Scope

Reach-pose motion adaptation in the original Simulator/iraLIB. This is not scoring,
pickup learning, camera navigation or physical-robot validation. GVF is not part of
this experiment. The only candidate change is episode-length-balanced draws from
the original accepted-row reservoir. Retention, reward, four learned actions,
controller, physics, teacher corpus and acceptance gates remain unchanged.

All new fits and mass simulation execute on the approved private Colab server,
not the laptop. This study deliberately uses its pinned CPU runtime; the allocated
T4 does not make it GPU training. Runtime: Linux x64, Python3.12.3,
Torch2.8.0+cpu, NumPy2.3.5 and Node22.14.0.

## Reproduction prerequisites

- Preserve the exact original source/runtime identity, immutable7,856-row/128-world
  teacher corpus, explicit Windows-to-Linux recorded-data admission, and complete
  source/native/discarded/development exposure ledger. Do not rewrite producer
  metadata to imply cross-host numerical parity.
- The same-host original/current initialization and complete synthetic optimizer
  states match. Actual full4 versus2/resume4 native continuation also matches,
  including critics, targets, Adam, replay, RNG, history and actor export.
- The complete pre-edit/current optional-off64-update controls match every decoded
  numerical state and all66 actor files. This does not fix the separately recorded
  Linux mismatch against frozen Windows initialization goldens; full regression
  must not be described as passed.
- Configuration: two hidden256-wide layers,64 updates,8 measured-corpus warm-start
  updates,256 collection decisions/update budget,32 gradient steps/collection,
  replay capacity1,024, zero critic-only warmup, full-corpus equal-world anchor,
  `episode-balanced-reservoir-v1`, original four-dimensional residual actions.
  Exact remaining parameters and source hashes are in `trial-protocol.json`, not
  inferred from this summary. Never resume a run after changing its bound sources.

## Measured development pilot

Seed201, run `67cbb420-fade-4208-91a3-1da837ea397b`, fixed update64:

| Complete suite | Controller | Own initial actor | Learned | All-world effort-proxy reduction |
| --- | ---: | ---: | ---: | ---: |
| Original32 | 27/32 | 25/32 | 29/32 | 7.22347133912743% |
| Mixed64 | 42/64 | 42/64 | 44/64 | 5.544530716204277% |

Both have zero controller-success regressions and zero per-case contact
regressions. Effort is a simulator proxy, not calibrated electrical energy.
Policy SHA: `10c3ba92f13ce42b2c0d3a53424d53a72cb5fc997ec16792cfa8561df1aa3585`.
Initial SHA: `41bbad40e488b119ed20a6e78ade5c410470d6d37aac7a2277659cb7ddfdbfef`.

This pilot does **not** establish independent improvement or qualify a default
policy. All prior failed methods remain preserved; no favorable seed is promoted.

## Independent qualification

**Recovery status,16:35 UTC:** Colab expired and the reconnected runtime has none
of the original independent run files. The last observed completion/partial run
does not establish their later outcomes. Independent qualification is incomplete,
not a pass or measured gate failure. The versioned loss registry blocks a new
final from silently excluding this missing work. Recover actual exposures or a
verified source-bound conservative superset before any new final; do not simply
mark the registry resolved, omit old seeds or rerun the old trial unnoticed.

The final evaluator and first-seed selection contract are frozen before separately
training seeds136001/136002/136003. Each gets the same fixed64-update configuration
and bounded server runtime, with no checkpoint selection or budget extension.
Every seed must independently pass both complete development suites:

- Strictly more successes than both original controller and its own initial actor.
- No lost original-controller success and no per-case contact increase.
- At least5% mean all-world time **or** effort-proxy improvement under the existing
  acceptance implementation; no rounding into a pass.

Only after all three pass and all artifacts/failures are preserved may a newly
untouched128-case final be prepared:8short/16field/8full geometry families with
four physical variations each. Exclude the complete actual exposure ledger,
freeze all six exact initial/learned actor hashes, burn the commitment before
revealing cases, and retain every outcome. All three must independently pass;
seed136001 represents the method only after every final seed passes.

No independent pass is claimed from this interrupted evidence chain. No consumed final
is reused for fitting; no automatic final preparation, default-policy promotion,
paid service or publication is authorized by the study scripts.

## Preserved local evidence

The complete proof/control/pilot archive is in ignored
`.cache/episode-balanced-colab-pilot-20261016.zip`,258,949,043 bytes, SHA
`c53261df5f2a049f36b7018973e801357c328b67009264f633eb08fcfa6cf922`.
All351 member hashes/CRCs verify.241 study files are imported under
`roboproof/runs/motion/research/episode-balanced-remote-study-20261016` without
overwriting current sources. These large artifacts are not implicitly published
or included in Git. Preserve them when moving to another computer.

The user-approved private Drive backup additionally contains four results-only
volumes covering all255 checkpoint/log/result/provenance files. Computer hashes
and CRCs verify; Drive names, sizes, destination and owner-only permissions verify.
The connector does not expose remote SHA readback, so that stronger check is not
claimed. No source-code files or credentials are uploaded to that folder. See
[remote custody requirements](REMOTE_TRAINING_CUSTODY.md) before further fits.
