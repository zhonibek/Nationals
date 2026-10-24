#!/usr/bin/env bash
# Creates the single bounded Nebius Serverless AI job that runs tools/nebius-tactical-job.sh.
# Runs where the Nebius CLI runs (Ubuntu, macOS or WSL), from a checkout of this repository.
#
# Without --launch it only validates the request (`nebius ai job create --dry-run`) and creates nothing.
# Use --launch only after the user has approved the GPU type, the maximum cost and the credits in chat.
# This script never buys credits, changes billing or enables automatic top-ups.
set -Eeuo pipefail

readonly BRANCH="nvidia-nebius-global-ai-hackathon"
readonly JOB_NAME="roboproof-tactical-seed731201"
readonly IMAGE="pytorch/pytorch@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385"  # 2.8.0-cuda12.8-cudnn9-runtime, linux/amd64
readonly PLATFORM="gpu-l40s-a" PRESET="1gpu-8vcpu-32gb" TIMEOUT="1h" DISK="50Gi"
# Worst case is the 1 h timeout on demand (Nebius Compute prices, October 2026): L40S Intel GPU $1.35
# + 8 vCPU x $0.012 + 32 GiB x $0.0032 = $1.548/h, plus 50 GiB network SSD at $0.071 per GiB-month ~ $0.005.
readonly CEILING_USD="1.60" MAX_APPROVABLE_USD="10"
readonly JOB_SCRIPT="tools/nebius-tactical-job.sh"

usage() {
  echo "usage: $0 --commit <sha> --bucket <private-bucket> --approved-max-cost-usd <usd> [--subnet-id <id>] [--launch]" >&2
  exit 64
}

commit="" bucket="" cost="" subnet="" launch=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --commit) commit=${2:-}; shift 2 ;;
    --bucket) bucket=${2:-}; shift 2 ;;
    --approved-max-cost-usd) cost=${2:-}; shift 2 ;;
    --subnet-id) subnet=${2:-}; shift 2 ;;
    --launch) launch=1; shift ;;
    *) usage ;;
  esac
done
[[ $commit =~ ^[0-9a-f]{40}$ && -n $bucket && $cost =~ ^[0-9]+(\.[0-9]+)?$ ]] || usage
awk -v cost="$cost" -v low="$CEILING_USD" -v high="$MAX_APPROVABLE_USD" 'BEGIN { exit !(cost >= low && cost <= high) }' ||
  { echo "approved maximum cost must cover the \$$CEILING_USD timeout ceiling and stay at most \$$MAX_APPROVABLE_USD" >&2; exit 65; }
command -v nebius > /dev/null || { echo "Nebius CLI not found (https://docs.nebius.com/cli/install)" >&2; exit 69; }

cd "$(git rev-parse --show-toplevel)"
git fetch --quiet origin "$BRANCH"
git merge-base --is-ancestor "$commit" "origin/$BRANCH" ||
  { echo "commit $commit is not pushed to origin/$BRANCH" >&2; exit 65; }
cmp -s "$JOB_SCRIPT" <(git show "$commit:$JOB_SCRIPT") ||
  { echo "$JOB_SCRIPT differs from the pinned commit" >&2; exit 65; }

arguments=(ai job create --name "$JOB_NAME" --image "$IMAGE" --platform "$PLATFORM" --preset "$PRESET"
  --on-demand --timeout "$TIMEOUT" --restart-policy never --disk-size "$DISK"
  --volume "$bucket:/output:rw" --inject-file "$JOB_SCRIPT:/opt/roboproof/nebius-tactical-job.sh"
  --env "ROBOPROOF_EXPECTED_COMMIT=$commit" --env "ROBOPROOF_APPROVED_MAX_COST_USD=$cost"
  --env "ROBOPROOF_IMAGE=$IMAGE" --env "ROBOPROOF_PLATFORM=$PLATFORM" --env "ROBOPROOF_PRESET=$PRESET"
  --container-command bash --args /opt/roboproof/nebius-tactical-job.sh)
[[ -z $subnet ]] || arguments+=(--subnet-id "$subnet")
if [[ $launch -eq 0 ]]; then
  arguments+=(--dry-run)
  echo "Validating only (no job is created). Approved ceiling: \$$cost; timeout ceiling: \$$CEILING_USD."
else
  echo "Creating ONE on-demand $PLATFORM/$PRESET job, timeout $TIMEOUT, approved ceiling \$$cost."
fi
printf '+ nebius'; printf ' %q' "${arguments[@]}"; printf '\n'
nebius "${arguments[@]}"
if [[ $launch -eq 1 ]]; then
  cat <<EOF
Next (read-only):
  nebius ai job get-by-name --name $JOB_NAME
  nebius ai job logs <job_ID> --follow --timestamps
Do not create a second job for seed 731201. Results: bucket $bucket, folder tactical-nebius-l40s-seed731201/.
EOF
fi
