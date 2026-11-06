#!/usr/bin/env bash
# Parallel tactical-training campaign on one AMD Instinct MI300X (AMD Developer Cloud droplet).
#
#   tools/amd-tactical-campaign.sh --hourly-rate-usd 1.99 --approved-max-cost-usd 20            # dry run: plan + checks
#   tools/amd-tactical-campaign.sh --launch --hourly-rate-usd 1.99 --approved-max-cost-usd 20   # start the runs
#
# Defaults: 34 runs (the most that fit in MI300X memory), PPO, v3 features, 1024 x 10 model, loader-cycle warm start, 2 h of training per run plus a 1 h
# evaluation reserve. Every run is predeclared in campaign.json (seeds checked against consumed_seeds.json) before
# anything starts, and each run is killed at the wall-clock limit so the cost ceiling holds. Run it inside tmux.
# Other planner options (--runs, --episodes, --max-seconds, --width, --hidden-layers, ...) are passed through.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-python3}
LAUNCH=0
OUTPUT_ROOT=""
ARGS=()
while (($#)); do
  case "$1" in
    --launch) LAUNCH=1; shift ;;
    --output-root) OUTPUT_ROOT=$2; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
OUTPUT_ROOT=${OUTPUT_ROOT:-roboproof/runs/tactical/mi300x-campaign-$(date -u +%Y%m%dT%H%M%SZ)}

if ((LAUNCH == 0)); then
  "$PYTHON" -m roboproof.tactical_learning.campaign plan --output-root "$OUTPUT_ROOT" --python "$PYTHON" "${ARGS[@]}"
  echo "Dry run only: nothing was started. Add --launch on the MI300X droplet to start the campaign."
  exit 0
fi

# --- launch checks: real MI300X under ROCm, Node.js, and sources that match the pinned commit ---
command -v node >/dev/null || { echo "Node.js 20+ is required" >&2; exit 1; }
command -v timeout >/dev/null || { echo "coreutils timeout is required" >&2; exit 1; }
"$PYTHON" - <<'PY'
import torch
assert torch.version.hip is not None and torch.cuda.is_available(), "a ROCm build of PyTorch with a visible GPU is required"
name = torch.cuda.get_device_properties(0).name
assert "MI300X" in name.upper(), f"expected an AMD Instinct MI300X, found {name}"
print(f"GPU check passed: {name}, HIP {torch.version.hip}, {torch.cuda.device_count()} visible")
PY
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "The working tree has uncommitted changes; run the campaign from a clean, pinned commit" >&2
  exit 1
fi

"$PYTHON" -m roboproof.tactical_learning.campaign plan --output-root "$OUTPUT_ROOT" --python "$PYTHON" --write "${ARGS[@]}"
WALL=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['wallClockLimitSeconds'])" "$OUTPUT_ROOT/campaign.json")
{
  echo "commit $(git rev-parse HEAD)"
  echo "started $(date -u +%FT%TZ)"
  echo "cpus $(nproc)"
  free -g || true
  (rocm-smi --showproductname || amd-smi static) 2>/dev/null || true
} > "$OUTPUT_ROOT/host.txt"

"$PYTHON" -m roboproof.tactical_learning.campaign sample "$OUTPUT_ROOT/gpu-utilization.jsonl" --interval 15 &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT

# One CPU thread per trainer: the trainers plus their Node rules engines already share the droplet's cores.
# rocBLAS instead of hipBLASLt: measured 1.8 GB instead of 4.1 GB host memory per trainer on the MI300X droplet,
# which is what lets 64 runs fit in RAM.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_BLAS_PREFER_HIPBLASLT=${TORCH_BLAS_PREFER_HIPBLASLT:-0}
echo "env OMP_NUM_THREADS=$OMP_NUM_THREADS TORCH_BLAS_PREFER_HIPBLASLT=$TORCH_BLAS_PREFER_HIPBLASLT" >> "$OUTPUT_ROOT/host.txt"
mapfile -t COMMANDS < <("$PYTHON" - "$OUTPUT_ROOT/campaign.json" <<'PY'
import json, shlex, sys
for run in json.load(open(sys.argv[1]))["runs"]:
    print(shlex.join(run["command"]))
PY
)
PIDS=()
for index in "${!COMMANDS[@]}"; do
  name=$(printf 'run-%02d' "$index")
  (
    status=0
    timeout --kill-after=120 "$WALL" bash -c "${COMMANDS[$index]}" > "$OUTPUT_ROOT/$name.log" 2>&1 || status=$?
    echo "$status" > "$OUTPUT_ROOT/$name.exit-code"
  ) &
  PIDS+=($!)
done
echo "Started ${#PIDS[@]} runs; wall-clock limit ${WALL} s. Logs: $OUTPUT_ROOT/run-XX.log"
for pid in "${PIDS[@]}"; do wait "$pid" || true; done

kill "$SAMPLER" 2>/dev/null || true
echo "finished $(date -u +%FT%TZ)" >> "$OUTPUT_ROOT/host.txt"
"$PYTHON" -m roboproof.tactical_learning.campaign summarize "$OUTPUT_ROOT"
(cd "$OUTPUT_ROOT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
tar -czf "$OUTPUT_ROOT.tar.gz" -C "$(dirname "$OUTPUT_ROOT")" "$(basename "$OUTPUT_ROOT")"
sha256sum "$OUTPUT_ROOT.tar.gz" | tee "$OUTPUT_ROOT.tar.gz.sha256"
# Two small archives that are realistic to preserve when the full one (all checkpoints) is too large to copy:
# every log, record and evaluation without checkpoints, and every run's final model without optimizer state.
NAME=$(basename "$OUTPUT_ROOT")
(
  cd "$(dirname "$OUTPUT_ROOT")"
  tar -czf "$NAME.results.tar.gz" --exclude="*.pt" "$NAME"
  sha256sum "$NAME.results.tar.gz" > "$NAME.results.tar.gz.sha256"
  models=("$NAME"/run-*/model-final.pt)
  if [ -e "${models[0]}" ]; then
    tar -czf "$NAME.models.tar.gz" "${models[@]}" "$NAME"/run-*/run.json "$NAME"/campaign.json "$NAME"/summary.json "$NAME"/SHA256SUMS
    sha256sum "$NAME.models.tar.gz" > "$NAME.models.tar.gz.sha256"
  fi
)
ls -la "$OUTPUT_ROOT".*tar.gz
echo "Next: copy the .results and .models archives (and the full one if possible) with their .sha256 files, verify, upload to the private RoboProof Drive folder, then DESTROY the droplet to stop billing."
