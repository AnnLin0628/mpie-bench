#!/usr/bin/env bash
# Running in the background VLM Judge v1(Security of resume running after breakpoint: kill terminal / Ctrl-C Just run the same command later)
#
# usage:
#   bash run_score_vlm_v1.sh gemini-3-pro-image
#   bash run_score_vlm_v1.sh gpt-image-2 8
#   PACK="$MPIE_TEST_PACK" JUDGE=gpt-5.5 bash run_score_vlm_v1.sh gemini-3-pro-image
set -euo pipefail

MODEL_ID="${1:?usage: $0 <model_id> [workers]}"
WORKERS="${2:-8}"
PACK="${PACK:-$HOME/mpie_testset_pack}"
JUDGE="${JUDGE:-gpt-5.5}"
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${PACK}/judgments/vlm_judge_v1/${MODEL_ID}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/_run_${JUDGE//\//_}_${STAMP}.log"

# gateway key: Environment variables take precedence; otherwise, they are used closedsource/gateway.py Internal source default
export AI_GATEWAY_KEY="${AI_GATEWAY_KEY:-${MPIE_VLM_KEY:-${AI_GATEWAY_KEY:-}}}"

PYTHON="${PYTHON:-$HOME/miniconda3/bin/python3}"
cd "$EVAL_DIR"
nohup "$PYTHON" -u score_vlm_v1.py \
  --pack "$PACK" \
  --model-id "$MODEL_ID" \
  --judge-model "$JUDGE" \
  --workers "$WORKERS" \
  --resume \
  >"$LOG" 2>&1 &

PID=$!
echo "started pid=$PID"
echo "log=$LOG"
echo "tail -f $LOG"
echo "resume: re-run this same command (skips finished JSON)"
echo "$PID" >"${LOG_DIR}/_run.pid"
