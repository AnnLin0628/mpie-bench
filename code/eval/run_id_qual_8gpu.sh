#!/usr/bin/env bash
# multi-GPU: 8 Card parallel running smoke100 of ArcFace ID + HPSv2 Qual
#
# Strategy:
#   - default MODE=by_model：5 Each model accounts for 1 Card(GPU0–4），ArcFace and HPS divided into two waves
#   - MODE=shard: Used for each model 8 Cards are sharded by sample (suitable for rush work on a single model)
#
# usage:
#   bash run_id_qual_8gpu.sh              # ArcFace full model 8 Card(by_model）
#   bash run_id_qual_8gpu.sh hps          # Run again HPSv2
#   bash run_id_qual_8gpu.sh both         # First ID back Qual
#   MODE=shard MODEL=gpt-image-2 bash run_id_qual_8gpu.sh arcface
set -euo pipefail

PACK="${MPIE_TEST_PACK:-"$MPIE_TEST_PACK"}"
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK="${1:-arcface}"          # arcface | hps | both
MODE="${MODE:-by_model}"      # by_model | shard
NGPU="${NGPU:-8}"
LOGDIR="${LOGDIR:-/tmp/mpie_id_qual_logs}"
mkdir -p "$LOGDIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MPIE_EVAL_ENV:-mpie}"
cd "$EVAL_DIR"

# Automatically discover models with output; available MODELS cover
if [[ -n "${MODELS:-}" ]]; then
  read -r -a MODELS <<< "$MODELS"
else
  mapfile -t MODELS < <(python - <<PY
from pathlib import Path
from pack_io import count_outputs
pack = Path("$PACK")
out = pack / "outputs"
ids = sorted(p.name for p in out.iterdir() if p.is_dir() and not p.name.startswith("_") and count_outputs(pack, p.name) > 0)
print("\n".join(ids))
PY
)
fi

echo "PACK=$PACK  TASK=$TASK  MODE=$MODE  NGPU=$NGPU"
echo "MODELS=${MODELS[*]}"

run_arcface_by_model() {
  local i=0
  local pids=()
  for mid in "${MODELS[@]}"; do
    local gpu=$((i % NGPU))
    local log="$LOGDIR/arcface_${mid}_gpu${gpu}.log"
    echo "[launch] ArcFace $mid -> GPU $gpu  log=$log"
    CUDA_VISIBLE_DEVICES="$gpu" python -u score_arcface_v1.py \
      --pack "$PACK" --model-id "$mid" --ctx-id 0 \
      >"$log" 2>&1 &
    pids+=($!)
    i=$((i + 1))
    # If the number of models > NGPU, wait for one batch before sending the next batch
    if (( i % NGPU == 0 )); then
      wait "${pids[@]}" || true
      pids=()
    fi
  done
  if ((${#pids[@]})); then wait "${pids[@]}" || true; fi
  echo "[done] ArcFace by_model"
}

run_hps_by_model() {
  local i=0
  local pids=()
  for mid in "${MODELS[@]}"; do
    local gpu=$((i % NGPU))
    local log="$LOGDIR/hps_${mid}_gpu${gpu}.log"
    echo "[launch] HPSv2 $mid -> GPU $gpu  log=$log"
    CUDA_VISIBLE_DEVICES="$gpu" python -u score_hpsv2.py \
      --pack "$PACK" --model-id "$mid" \
      >"$log" 2>&1 &
    pids+=($!)
    i=$((i + 1))
    if (( i % NGPU == 0 )); then
      wait "${pids[@]}" || true
      pids=()
    fi
  done
  if ((${#pids[@]})); then wait "${pids[@]}" || true; fi
  echo "[done] HPSv2 by_model"
}

run_arcface_shard() {
  local mid="${MODEL:?MODE=shard need export MODEL=someone model_id}"
  local pids=()
  for gpu in $(seq 0 $((NGPU - 1))); do
    local log="$LOGDIR/arcface_${mid}_shard${gpu}.log"
    echo "[launch] ArcFace $mid shard $gpu/$NGPU -> GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python -u score_arcface_v1.py \
      --pack "$PACK" --model-id "$mid" --ctx-id 0 \
      --shard-id "$gpu" --num-shards "$NGPU" \
      >"$log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}" || true
  echo "[done] ArcFace shard $mid"
}

run_hps_shard() {
  local mid="${MODEL:?MODE=shard need export MODEL=someone model_id}"
  local pids=()
  for gpu in $(seq 0 $((NGPU - 1))); do
    local log="$LOGDIR/hps_${mid}_shard${gpu}.log"
    echo "[launch] HPSv2 $mid shard $gpu/$NGPU -> GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python -u score_hpsv2.py \
      --pack "$PACK" --model-id "$mid" \
      --shard-id "$gpu" --num-shards "$NGPU" \
      >"$log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}" || true
  echo "[done] HPSv2 shard $mid"
}

do_arcface() {
  if [[ "$MODE" == "shard" ]]; then run_arcface_shard; else run_arcface_by_model; fi
}
do_hps() {
  if [[ "$MODE" == "shard" ]]; then run_hps_shard; else run_hps_by_model; fi
}

case "$TASK" in
  arcface) do_arcface ;;
  hps|hpsv2) do_hps ;;
  both)
    do_arcface
    do_hps
    ;;
  *)
    echo "usage: $0 [arcface|hps|both]"; exit 1 ;;
esac

echo "---- coverage ----"
for m in "${MODELS[@]}"; do
  a=$(ls "$PACK/judgments/arcface_v1/$m"/*.json 2>/dev/null | grep -v _run_summary | wc -l)
  h=$(ls "$PACK/judgments/hpsv2/$m"/*.json 2>/dev/null | grep -v _run_summary | wc -l)
  o=$(ls "$PACK/outputs/$m"/*.png 2>/dev/null | wc -l)
  echo "$m  out=$o  arcface=$a  hpsv2=$h"
done
echo "logs: $LOGDIR"
