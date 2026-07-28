#!/usr/bin/env bash
# Build frozen Instr QA v2 bank (text-only; TIFA/MultiHuman style).
#
# Usage:
#   bash run_build_instr_qa_v2.sh
#   bash run_build_instr_qa_v2.sh 8
#   PACK="$MPIE_TEST_PACK" JUDGE=gpt-5.5 bash run_build_instr_qa_v2.sh 8
set -euo pipefail

WORKERS="${1:-8}"
PACK="${PACK:-$HOME/mpie_testset_pack}"
JUDGE="${JUDGE:-gpt-5.5}"
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${PACK}/instr_qa_v2"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/_build_${JUDGE//\//_}_${STAMP}.log"

if [[ -f "${HOME}/.mpie_env" ]]; then
  # shellcheck disable=SC1090
  set -a; source "${HOME}/.mpie_env"; set +a
fi
export AI_GATEWAY_KEY="${AI_GATEWAY_KEY:-${MPIE_VLM_KEY:-${AI_GATEWAY_KEY:-}}}"

PYTHON="${PYTHON:-$HOME/miniconda3/bin/python3}"
cd "$EVAL_DIR"
nohup "$PYTHON" -u build_instr_qa_v2.py \
  --pack "$PACK" \
  --qg-model "$JUDGE" \
  --workers "$WORKERS" \
  --filter \
  --resume \
  >"$LOG" 2>&1 &

PID=$!
echo "started pid=$PID"
echo "log=$LOG"
echo "tail -f $LOG"
echo "$PID" >"${LOG_DIR}/_build.pid"
