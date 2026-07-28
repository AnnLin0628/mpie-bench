#!/usr/bin/env bash
# Running in the background Checklist_V(Same as human beings Anat/Inter 0/1;Non-old six-axis v1）
# gateway with VQA(score_vlm_v1) same:gpt-5.5 + AI_GATEWAY_URL + AI_GATEWAY_KEY
#
# usage:
#   bash run_score_checklist_vlm.sh pilot
#   bash run_score_checklist_vlm.sh pilot,holdout 4
#   PACK="$MPIE_TEST_PACK" JUDGE=gpt-5.5 LIMIT=2 bash run_score_checklist_vlm.sh pilot
set -euo pipefail

SPLIT="${1:-pilot}"
WORKERS="${2:-4}"
PACK="${PACK:-$HOME/mpie_testset_pack}"
JUDGE="${JUDGE:-gpt-5.5}"
LIMIT="${LIMIT:-0}"
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${PACK}/judgments/human_consistency/checklist_vlm/${JUDGE//\//_}"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${OUT_DIR}/_run_${SPLIT//,/_}_${STAMP}.log"

# and closed source/VQA Same: load the gateway first env(Don’t put key (Photograph script)
for envf in \
  "${GATEWAY_ENV:-}" \
  your env file \
  /tmp/mpie_closed_full_env.sh \
  "${HOME}/.mpie_env"
do
  if [[ -n "${envf}" && -f "${envf}" ]]; then
    # shellcheck disable=SC1090
    set -a
    source "${envf}"
    set +a
  fi
done
export AI_GATEWAY_KEY="${AI_GATEWAY_KEY:-${MPIE_VLM_KEY:-${AI_GATEWAY_KEY:-${GPT_IMAGE_KEY:-}}}}"

if [[ -z "${AI_GATEWAY_KEY:-}" ]]; then
  echo "ERROR: Lack AI_GATEWAY_KEY(and VQA same). please first: set -a; source your env file; set +a"
  exit 1
fi
if [[ -z "${AI_GATEWAY_URL:-${MPIE_VLM_URL:-${MPIE_GATEWAY_URL:-}}}" ]]; then
  echo "ERROR: Lack AI_GATEWAY_URL(and VQA same)"
  exit 1
fi

PYTHON="${PYTHON:-$HOME/miniconda3/bin/python}"
cd "$EVAL_DIR"

EXTRA=()
if [[ "${LIMIT}" != "0" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi

nohup "$PYTHON" -u score_checklist_vlm.py \
  --pack "$PACK" \
  --split "$SPLIT" \
  --judge-model "$JUDGE" \
  --workers "$WORKERS" \
  --resume \
  "${EXTRA[@]}" \
  >"$LOG" 2>&1 &

PID=$!
echo "started pid=$PID"
echo "log=$LOG"
echo "judge=$JUDGE (same gateway as score_vlm_v1)"
echo "endpoint_env=AI_GATEWAY_URL set"
echo "tail -f $LOG"
echo "resume: re-run same command (skips finished JSON)"
echo "$PID" >"${OUT_DIR}/_run.pid"
