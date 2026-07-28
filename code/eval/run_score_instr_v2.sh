#!/usr/bin/env bash
# Score Instr v2 for one model (frozen QA + image-only answers).
#
# Usage:
#   bash run_score_instr_v2.sh gpt-image-2
#   bash run_score_instr_v2.sh gemini-3-pro-image 8
#   PACK="$MPIE_TEST_PACK" bash run_score_instr_v2.sh flux1-kontext-dev 8
#
# Requires: $PACK/instr_qa_v2/*.json  (run run_build_instr_qa_v2.sh first)
set -euo pipefail

MODEL_ID="${1:?usage: $0 <model_id> [workers]}"
WORKERS="${2:-8}"
PACK="${PACK:-$HOME/mpie_testset_pack}"
JUDGE="${JUDGE:-gpt-5.5}"
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${PACK}/judgments/instr_v2/${MODEL_ID}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/_run_${JUDGE//\//_}_${STAMP}.log"

if [[ -f "${HOME}/.mpie_env" ]]; then
  # shellcheck disable=SC1090
  set -a; source "${HOME}/.mpie_env"; set +a
fi
export AI_GATEWAY_KEY="${AI_GATEWAY_KEY:-${MPIE_VLM_KEY:-${AI_GATEWAY_KEY:-}}}"

if [[ ! -d "${PACK}/instr_qa_v2" ]] || [[ -z "$(ls -A "${PACK}/instr_qa_v2"/*.json 2>/dev/null || true)" ]]; then
  echo "ERROR: missing frozen QA bank at ${PACK}/instr_qa_v2/"
  echo "Run: bash run_build_instr_qa_v2.sh"
  exit 1
fi

PYTHON="${PYTHON:-$HOME/miniconda3/bin/python3}"
cd "$EVAL_DIR"
nohup "$PYTHON" -u score_instr_v2.py \
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
echo "$PID" >"${LOG_DIR}/_run.pid"
