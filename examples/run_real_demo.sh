#!/bin/sh
# Real-world dedrift demo: a live agent, a real model swap, real detection.
#
# 18 canaries x 5 repetitions x 6 cycles = 540 API calls (short outputs).
# Cycles 1-4 run MODEL_A; cycles 5-6 run MODEL_B — the "provider swapped
# the model under you" scenario, for real.
#
# Usage:
#   export ANTHROPIC_API_KEY=...
#   sh examples/run_real_demo.sh [demo-dir]
set -eu

DIR="${1:-real-demo}"
MODEL_A="${MODEL_A:-claude-haiku-4-5}"
MODEL_B="${MODEL_B:-claude-sonnet-5}"
REPS="${REPS:-5}"
SUITE="$(cd "$(dirname "$0")" && pwd)/canaries_real.yaml"
AGENT="examples.anthropic_agent:agent_fn"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "$DIR" && cd "$DIR"

dedrift init
dedrift embedder pin hash

echo ">>> Baseline era: 4 cycles on $MODEL_A"
export DEDRIFT_DEMO_MODEL="$MODEL_A"
dedrift canary run --suite "$SUITE" --agent "$AGENT" \
    --model "anthropic/$MODEL_A" --repetitions "$REPS" --cycles 4

echo ">>> Freezing the first 3 cycles as the golden baseline"
dedrift baseline set --first 3

echo ">>> The silent swap: 2 cycles on $MODEL_B (same prompt, same canaries)"
export DEDRIFT_DEMO_MODEL="$MODEL_B"
dedrift canary run --suite "$SUITE" --agent "$AGENT" \
    --model "anthropic/$MODEL_B" --repetitions "$REPS" --cycles 2

echo ">>> Did dedrift notice?"
dedrift check || true
dedrift report --out report.md
echo ">>> Report written to $DIR/report.md"
