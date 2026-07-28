#!/usr/bin/env bash
# Human-consistent annotation front-end (8080）
#
# usage:
#   bash run_annot_frontend.sh
#   PACK="$MPIE_TEST_PACK" PORT=8080 bash run_annot_frontend.sh
set -euo pipefail

PACK="${PACK:-$HOME/mpie_testset_pack}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${PACK}/judgments/human_consistency/annot_frontend_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/_run_${STAMP}.log"
PYTHON="${PYTHON:-$HOME/miniconda3/bin/python}"

# Release the old process (only for this port) app_8080）
if command -v ss >/dev/null 2>&1; then
  OLD_PID="$(ss -ltnp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {print}' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1 || true)"
  if [[ -n "${OLD_PID:-}" ]]; then
    echo "killing old pid=$OLD_PID on :$PORT"
    kill "$OLD_PID" 2>/dev/null || true
    sleep 0.4
  fi
fi

cd "$EVAL_DIR/annot_frontend"
export PACK PORT HOST
nohup "$PYTHON" -u app.py >"$LOG" 2>&1 &
PID=$!
echo "$PID" >"${LOG_DIR}/_run.pid"
sleep 0.8
echo "started pid=$PID"
echo "url=http://127.0.0.1:${PORT}/"
echo "annotators=ann_01 ann_02 ann_03 ann_04 ann_05 ann_06 (need 6)"
echo "log=$LOG"
echo "pack=$PACK"
