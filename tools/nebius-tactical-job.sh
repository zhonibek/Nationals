#!/usr/bin/env bash
# One bounded tactical actor-critic fit inside a Nebius AI Cloud Serverless AI job (L40S).
# Create the job only with tools/nebius-tactical-launch.sh, after the user approves the GPU type,
# maximum cost and credits. Runs as the container entrypoint; the results bucket is mounted at /output.
#
# The fit is predeclared here, so it is bound to the pinned commit. Seed 731201 may be used once:
# a new fit needs a new committed seed, never a rerun of this script.
set -Eeuo pipefail

readonly REPO_URL="https://github.com/zhonibek/Nationals.git"
readonly BRANCH="nvidia-nebius-global-ai-hackathon"
readonly SEED=731201 EPISODES=100 MAX_SECONDS=600 EVALUATION_EPISODES=8 GPU="L40S"
readonly NODE_VERSION="v20.19.0"
readonly NODE_SHA256="8a4dbcdd8bccef3132d21e8543940557e55dcf44f00f0a99ba8a062f4552e722"  # node-v20.19.0-linux-x64.tar.gz
readonly RUN_NAME="tactical-nebius-l40s-seed${SEED}"

log() { printf '[roboproof %s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
fail() { printf '[roboproof] FAILED: %s\n' "$*" >&2; exit 2; }

# Hashes every file of <parent>/<name> into SHA256SUMS, writes <out>/<name>.tar.gz plus its .sha256,
# and prints the archive SHA-256.
package_run() {
  local parent=$1 name=$2 out=$3 digest
  (cd "$parent" && find "$name" -type f ! -path "$name/SHA256SUMS" -print0 | LC_ALL=C sort -z |
    xargs -0 -r sha256sum) > "$parent/$name/SHA256SUMS"
  tar --sort=name --owner=0 --group=0 --numeric-owner --mtime=@0 -C "$parent" -cf - "$name" | gzip -n > "$out/$name.tar.gz"
  digest=$(sha256sum "$out/$name.tar.gz" | cut -d' ' -f1)
  printf '%s  %s.tar.gz\n' "$digest" "$name" > "$out/$name.tar.gz.sha256"
  printf '%s\n' "$digest"
}

write_job_record() {
  JOB_RECORD="$STAGE/job.json" python - <<'PY'
import json, os
environment = os.environ
record = {"schemaVersion": 1, "provider": "nebius-serverless-ai-job", "runName": environment["RUN_NAME"],
          "repository": environment["REPO_URL"], "branch": environment["BRANCH"],
          "branchHead": environment.get("BRANCH_HEAD") or None, "commit": environment["ROBOPROOF_EXPECTED_COMMIT"],
          "declaredImage": environment.get("ROBOPROOF_IMAGE"), "declaredPlatform": environment.get("ROBOPROOF_PLATFORM"),
          "declaredPreset": environment.get("ROBOPROOF_PRESET"), "approvedGpu": environment["GPU"],
          "approvedMaxCostUsd": float(environment["ROBOPROOF_APPROVED_MAX_COST_USD"]), "providerBillingVerified": False,
          "declaredPaymentSource": environment.get("ROBOPROOF_PAYMENT_SOURCE"),
          "declaredFreeCreditBalanceUsd": environment.get("ROBOPROOF_FREE_CREDIT_BALANCE_USD"),
          "seed": int(environment["SEED"]), "episodes": int(environment["EPISODES"]),
          "maxSeconds": int(environment["MAX_SECONDS"]), "evaluationEpisodes": int(environment["EVALUATION_EPISODES"]),
          "node": {"version": environment["NODE_VERSION"], "sha256": environment["NODE_SHA256"]},
          "startedAt": environment["STARTED_AT"], "finishedAt": environment.get("FINISHED_AT") or None,
          "trainerExitStatus": int(environment["TRAINER_EXIT"]) if environment.get("TRAINER_EXIT") else None,
          "jobExitStatus": int(environment["JOB_EXIT"]) if environment.get("JOB_EXIT") else None,
          "learnedImprovementVerified": False, "independentQualificationPerformed": False}
with open(environment["JOB_RECORD"], "x", encoding="utf-8") as handle:
    handle.write(json.dumps(record, indent=2) + "\n")
PY
}

# EXIT trap: package whatever exists (including failed or partial runs) and copy it to the bucket.
preserve() {
  JOB_EXIT=$?
  set +e
  trap - EXIT
  FINISHED_AT=$(date -u +%FT%TZ)
  export JOB_EXIT FINISHED_AT TRAINER_EXIT BRANCH_HEAD
  write_job_record || log "job.json could not be written"
  local digest
  digest=$(package_run "$STAGE_PARENT" "$RUN_NAME" "$WORK/archive")
  log "manifest $RUN_NAME/SHA256SUMS:"
  cat "$STAGE/SHA256SUMS"
  log "archive sha256 $digest  $RUN_NAME.tar.gz ($(stat -c %s "$WORK/archive/$RUN_NAME.tar.gz") bytes)"
  [[ ! -f $STAGE/job.json ]] || cp "$STAGE/job.json" "$DEST/"
  if cp "$WORK/archive/$RUN_NAME.tar.gz" "$WORK/archive/$RUN_NAME.tar.gz.sha256" "$STAGE/SHA256SUMS" "$DEST/" &&
      (cd "$DEST" && sha256sum -c --quiet "$RUN_NAME.tar.gz.sha256"); then
    log "custody: archive copied to the results bucket and read back through the mount with a matching SHA-256"
  else
    log "custody: COPY TO THE RESULTS BUCKET FAILED; the job logs above are the only record"
    [[ $JOB_EXIT -ne 0 ]] || JOB_EXIT=3
  fi
  log "finished with exit status $JOB_EXIT"
  exit "$JOB_EXIT"
}

main() {
  [[ ${ROBOPROOF_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || fail "ROBOPROOF_EXPECTED_COMMIT must be a full commit SHA"
  [[ ${ROBOPROOF_APPROVED_MAX_COST_USD:-} =~ ^[0-9]+(\.[0-9]+)?$ ]] || fail "ROBOPROOF_APPROVED_MAX_COST_USD is required"
  OUTPUT_ROOT=${ROBOPROOF_OUTPUT_ROOT:-/output}
  WORK=${ROBOPROOF_WORK_ROOT:-/workspace/roboproof-job}
  DEST="$OUTPUT_ROOT/$RUN_NAME" STAGE_PARENT="$WORK/stage" STAGE="$WORK/stage/$RUN_NAME"
  [[ -d $OUTPUT_ROOT && -w $OUTPUT_ROOT ]] || fail "no writable results bucket at $OUTPUT_ROOT; nothing ran"
  [[ ! -e $DEST ]] || fail "$DEST exists: seed $SEED was already claimed; never rerun it"
  [[ ! -e $WORK ]] || fail "$WORK already exists"
  mkdir -p "$STAGE" "$WORK/archive"
  STARTED_AT=$(date -u +%FT%TZ) TRAINER_EXIT="" BRANCH_HEAD=""
  export RUN_NAME REPO_URL BRANCH GPU SEED EPISODES MAX_SECONDS EVALUATION_EPISODES NODE_VERSION NODE_SHA256 STARTED_AT
  # From here on the seed counts as claimed; the EXIT trap preserves every outcome.
  mkdir "$DEST"
  trap preserve EXIT
  printf '{"runName":"%s","seed":%d,"commit":"%s","claimedAt":"%s"}\n' \
    "$RUN_NAME" "$SEED" "$ROBOPROOF_EXPECTED_COMMIT" "$STARTED_AT" > "$DEST/claimed.json"

  log "GPU allocation as seen inside the container:"
  nvidia-smi 2>&1 | tee "$STAGE/nvidia-smi.txt"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$STAGE/gpu.csv"

  log "installing git and curl"
  export DEBIAN_FRONTEND=noninteractive
  { apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates; } > "$STAGE/apt.log" 2>&1 ||
    fail "apt-get failed (see apt.log)"

  log "cloning $BRANCH and pinning $ROBOPROOF_EXPECTED_COMMIT"
  git clone --quiet --single-branch --branch "$BRANCH" "$REPO_URL" "$WORK/Nationals"
  BRANCH_HEAD=$(git -C "$WORK/Nationals" rev-parse HEAD)
  git -C "$WORK/Nationals" merge-base --is-ancestor "$ROBOPROOF_EXPECTED_COMMIT" HEAD ||
    fail "commit $ROBOPROOF_EXPECTED_COMMIT is not on $BRANCH"
  git -C "$WORK/Nationals" checkout --quiet --detach "$ROBOPROOF_EXPECTED_COMMIT"
  [[ $(sha256sum < "${BASH_SOURCE[0]}") == $(sha256sum < "$WORK/Nationals/tools/nebius-tactical-job.sh") ]] ||
    fail "the injected job script differs from tools/nebius-tactical-job.sh at the pinned commit"

  log "installing Node.js $NODE_VERSION (SHA-256 pinned)"
  curl -fsSL -o "$WORK/node.tar.gz" "https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-linux-x64.tar.gz"
  echo "$NODE_SHA256  $WORK/node.tar.gz" | sha256sum -c --quiet - || fail "Node.js archive SHA-256 mismatch"
  mkdir "$WORK/node" && tar -xzf "$WORK/node.tar.gz" -C "$WORK/node" --strip-components=1
  export PATH="$WORK/node/bin:$PATH"

  log "training: seed $SEED, $EPISODES episodes, at most $MAX_SECONDS s, $EVALUATION_EPISODES evaluation episodes"
  cd "$WORK/Nationals"
  set +e
  python -u -m roboproof.tactical_learning.train --approved-nebius-gpu "$GPU" \
    --approved-max-cost-usd "$ROBOPROOF_APPROVED_MAX_COST_USD" --seed "$SEED" --episodes "$EPISODES" \
    --max-seconds "$MAX_SECONDS" --evaluation-episodes "$EVALUATION_EPISODES" --output "$STAGE/run" 2>&1 |
    tee "$STAGE/stdout.txt"
  TRAINER_EXIT=${PIPESTATUS[0]}
  set -e
  log "trainer exit status $TRAINER_EXIT"
  if [[ -f $STAGE/run/evaluation.json ]]; then
    python - "$STAGE/run/evaluation.json" <<'PY'
import json, sys
evaluation = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({"episodes": len(evaluation["episodes"]), "meanScoreMargin": evaluation["meanScoreMargin"],
                  "baselineMeanScoreMargin": evaluation["baselineMeanScoreMargin"],
                  "beatsLoaderCycleOnHoldout": evaluation["beatsLoaderCycleOnHoldout"]}, sort_keys=True))
PY
  fi
  return "$TRAINER_EXIT"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
