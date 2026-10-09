#!/bin/sh
# Optional live-provider experiment: compare two deliberately different models.
# This is not a production validation and a quiet result is possible.
#
# 18 canaries x the default 7 repetitions x 6 cycles = 756 API calls.
# Cycles 1-4 run MODEL_A; cycles 5-6 run MODEL_B — the "provider swapped
# the model under you" scenario, deliberately simulated with an explicit swap.
#
# Usage (API key):
#   export ANTHROPIC_API_KEY=...
#   sh examples/run_real_demo.sh [demo-dir]
#
# Usage (Claude subscription, via the Claude Code CLI, logged in):
#   AGENT=examples.claude_code_agent:agent_fn MODEL_A=haiku MODEL_B=sonnet \
#     sh examples/run_real_demo.sh [demo-dir]
set -eu

DIR="${1:-real-demo}"
MODEL_A="${MODEL_A:-claude-haiku-4-5}"
MODEL_B="${MODEL_B:-claude-sonnet-5}"
SUITE="$(cd "$(dirname "$0")" && pwd)/canaries_real.yaml"
AGENT="${AGENT:-examples.anthropic_agent:agent_fn}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "$DIR" && cd "$DIR"
REPORT_PATH="$(pwd)/report.md"

dedrift init
dedrift embedder pin hash

echo ">>> Baseline era: 4 cycles on $MODEL_A"
export DEDRIFT_DEMO_MODEL="$MODEL_A"
dedrift canary run --suite "$SUITE" --agent "$AGENT" \
    --model "anthropic/$MODEL_A" --cycles 4

echo ">>> Freezing the first 3 cycles as the golden baseline"
dedrift baseline set --first 3

echo ">>> The silent swap: 2 cycles on $MODEL_B (same prompt, same canaries)"
export DEDRIFT_DEMO_MODEL="$MODEL_B"
dedrift canary run --suite "$SUITE" --agent "$AGENT" \
    --model "anthropic/$MODEL_B" --cycles 2

echo ">>> What did dedrift measure? (2=drift, 3=inconclusive, 1=operational failure)"
set +e
dedrift check
CHECK_STATUS=$?
set -e
case "$CHECK_STATUS" in
    0|2|3) ;;
    *) exit "$CHECK_STATUS" ;;
esac

# report runs the check again and intentionally preserves the same semantic
# 0/2/3 exit codes. Do not use `dedrift check && dedrift report`: a valid
# drift result exits 2 and would skip the report.
set +e
dedrift report --out "$REPORT_PATH"
REPORT_STATUS=$?
set -e
case "$REPORT_STATUS" in
    0|2|3) ;;
    *) exit "$REPORT_STATUS" ;;
esac

echo ">>> Check status: $CHECK_STATUS; report status: $REPORT_STATUS"
echo ">>> Report written to $REPORT_PATH"
