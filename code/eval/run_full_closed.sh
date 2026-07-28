#!/usr/bin/env bash
# Whole amount pack Closed Source Picture of Three Family Students (gpt-image-2 / gemini-3-pro-image(=nano-banana-pro) / seedream-5-pro）
#
# Resume running from breakpoint:run_closed.py according to outputs/<model_id>/<sample_id>.png Whether there is already a skip;
# Change to cheaper one API Then, rerun with the same command to only fill in the missing samples.
#
# usage:
#   # Prepare the environment first (do not put key (Photograph script)
#   set -a; source /tmp/mpie_closed_full_env.sh; set +a
#   # or by oneself export:
#   #   AI_GATEWAY_KEY / AI_GATEWAY_URL
#   #   SEEDREAM_KEY / SEEDREAM_URL / SEEDREAM_MODEL
#
#   bash "$MPIE_ROOT/code/eval/run_full_closed.sh
#   bash "$MPIE_ROOT/code/eval/run_full_closed.sh --limit 2          # Ahead aisle
#   MODELS="gpt-image-2" bash "$MPIE_ROOT/code/eval/run_full_closed.sh
#   WORKERS_GPT=4 WORKERS_GEM=4 WORKERS_SEED=6 bash ...
set -euo pipefail

PACK="${MPIE_TEST_PACK:-$HOME/mpie_testset_pack}"
CODE="${MPIE_BENCH:-$HOME/mpie_bench}/code/eval/closedsource"
MODELS="${MODELS:-gpt-image-2 gemini-3-pro-image seedream-5-pro}"
WORKERS_GPT="${WORKERS_GPT:-4}"
WORKERS_GEM="${WORKERS_GEM:-4}"
WORKERS_SEED="${WORKERS_SEED:-6}"
EXTRA=("$@")

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$HOME/miniconda3/envs/pipeline/bin/python" ]]; then
  PY="$HOME/miniconda3/envs/pipeline/bin/python"
elif [[ -x "$HOME/miniconda3/bin/python" ]]; then
  PY="$HOME/miniconda3/bin/python"
else
  PY="python3"
fi

if [[ ! -f "$PACK/manifest.jsonl" ]]; then
  echo "ERROR: missing $PACK/manifest.jsonl"
  exit 1
fi

need_gateway=0
need_seedream=0
for m in $MODELS; do
  case "$m" in
    gpt-image-2|gemini-3-pro-image|nano-banana-pro) need_gateway=1 ;;
    seedream-5-pro|seedream-5) need_seedream=1 ;;
  esac
done
if [[ $need_gateway -eq 1 ]]; then
  if [[ -z "${AI_GATEWAY_KEY:-${AI_GATEWAY_KEY:-}}" || -z "${AI_GATEWAY_URL:-${MPIE_GATEWAY_URL:-}}" ]]; then
    echo "ERROR: need AI_GATEWAY_KEY + AI_GATEWAY_URL"
    exit 1
  fi
fi
if [[ $need_seedream -eq 1 ]]; then
  if [[ -z "${SEEDREAM_KEY:-${SEEDREAM5_LITE_KEY:-}}" ]]; then
    echo "ERROR: need SEEDREAM_KEY or SEEDREAM5_LITE_KEY"
    exit 1
  fi
fi

N=$(wc -l < "$PACK/manifest.jsonl")
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PACK/outputs"
echo "PACK=$PACK samples=$N stamp=$STAMP models=[$MODELS]"
echo "resume: skip existing outputs/<model_id>/*.png (>1KB)"

cd "$CODE"
PIDS=()
for m in $MODELS; do
  case "$m" in
    gpt-image-2) W="$WORKERS_GPT" ;;
    gemini-3-pro-image|nano-banana-pro) W="$WORKERS_GEM" ;;
    seedream-5-pro|seedream-5) W="$WORKERS_SEED" ;;
    *) W=4 ;;
  esac
  LOG="$PACK/outputs/_run_${m//\//_}_full_${STAMP}.log"
  echo "start $m workers=$W log=$LOG"
  nohup "$PY" -u run_closed.py --model "$m" --pack "$PACK" --workers "$W" "${EXTRA[@]}" \
    >"$LOG" 2>&1 &
  echo $! >"$PACK/outputs/_run_${m//\//_}_full.pid"
  PIDS+=($!)
done

echo "launched pids: ${PIDS[*]}"
echo "monitor:"
echo "  for m in $MODELS; do echo -n \"\$m \"; ls \"$PACK/outputs/\$m\"/*.png 2>/dev/null | wc -l; done"
echo "  tail -f $PACK/outputs/_run_*_full_${STAMP}.log"
