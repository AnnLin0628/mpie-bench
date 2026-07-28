#!/bin/bash
# Full open source × Multi-card sample sharding
#
# Required/Commonly used environment variables:
#   export MPIE_TEST_PACK=/path/to/mpie_testset_pack
#   export MPIE_WEIGHTS=/path/to/mpie_weights
#   export MPIE_BENCH_EVAL=/path/to/mpie_bench/code/eval/opensource   # Optional
#   LIMIT=2 NGPU=1 bash run_opensource_full_8gpu.sh kontext          # smoke
#   NGPU=8 bash run_opensource_full_8gpu.sh kontext                  # Whole amount
#
# Shared disk/FUSE Do not run multiple processes at the same time mmap Loading; Available:
#   HIGH_VRAM=1 NGPU=8 bash run_opensource_full_8gpu.sh kontext
# （HIGH_VRAM=1 default STAGGER_READY_RE='pipe ready'）
# See details docs/02_pipeline_design/eval_opensource.md
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PACK="${MPIE_TEST_PACK:-${HOME}/mpie_testset_pack}"
CODE="${MPIE_BENCH_EVAL:-${SCRIPT_DIR}/opensource}"
export MPIE_WEIGHTS="${MPIE_WEIGHTS:-${HOME}/mpie_weights}"
export MPIE_CODE="${MPIE_CODE:-$HOME/mpie_code}"

# HIGH_VRAM=1：kontext Not on fp8, the entire mold is loaded;qwen use --offload none;Default staggered loading
HIGH_VRAM="${HIGH_VRAM:-0}"
NGPU="${NGPU:-8}"
LIMIT="${LIMIT:-0}"
# many seed experiment:SEED=1 SEED_TAG=1 → outputs/<model>_s1/ + runner --seed 1 --seed-tag
SEED="${SEED:-}"
SEED_TAG="${SEED_TAG:-0}"
# physics GPU List (jump bad cards). example:GPUS=0,1,2,4,5,6,7 → 7 Road fragmentation, skipping GPU3
# Overwrite after setting NGPU(number of shards = list length)
GPUS="${GPUS:-}"
# Off-peak start:
# - Shared disk/FUSE:avoid N road at the same time mmap(See eval_opensource.md）
# - DreamO int8:avoid N The road is quantitatively exploded at the same time RAM
# STAGGER_SEC>0: one for each shard back sleep N Start the next one in seconds
# STAGGER_READY_RE: Wait until the shard The log matches the regular pattern (or times out) and then starts the next one.
STAGGER_SEC="${STAGGER_SEC:-0}"
STAGGER_READY_RE="${STAGGER_READY_RE:-}"
STAGGER_READY_TIMEOUT="${STAGGER_READY_TIMEOUT:-1800}"
if [[ "$HIGH_VRAM" == "1" && -z "$STAGGER_READY_RE" && "$STAGGER_SEC" -eq 0 ]]; then
  STAGGER_READY_RE='pipe ready'
fi
LOGDIR="${LOGDIR:-/tmp/mpie_opensource_full_logs}"
mkdir -p "$LOGDIR"

if [[ ! -f "$PACK/manifest.jsonl" ]]; then
  echo "ERROR: missing $PACK/manifest.jsonl"
  echo "  Set MPIE_TEST_PACK to a pack directory that contains manifest.jsonl"
  exit 1
fi
N=$(wc -l < "$PACK/manifest.jsonl")

# Resolve GPU list → GPU_ARR + NGPU
if [[ -n "$GPUS" ]]; then
  IFS=',' read -r -a GPU_ARR <<< "$GPUS"
  NGPU="${#GPU_ARR[@]}"
else
  GPU_ARR=()
  for ((g=0; g<NGPU; g++)); do GPU_ARR+=("$g"); done
fi

echo "PACK=$PACK samples=$N weights=$MPIE_WEIGHTS code=$CODE NGPU=$NGPU GPUS=[${GPU_ARR[*]}] HIGH_VRAM=$HIGH_VRAM stagger_re=${STAGGER_READY_RE:-none} stagger_sec=$STAGGER_SEC"
nvidia-smi -L | head -10

LIMIT_ARGS=()
if [[ "$LIMIT" != "0" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi
SEED_ARGS=()
if [[ -n "$SEED" ]]; then
  SEED_ARGS+=(--seed "$SEED")
fi
if [[ "$SEED_TAG" == "1" || "$SEED_TAG" == "true" || "$SEED_TAG" == "yes" ]]; then
  SEED_ARGS+=(--seed-tag)
fi
if ((${#SEED_ARGS[@]})); then
  echo "SEED_ARGS=${SEED_ARGS[*]}"
fi

wait_shard_ready() {
  # $1=log $2=pid $3=label
  local log="$1" pid="$2" label="$3"
  local t0=$SECONDS
  if [[ -z "$STAGGER_READY_RE" && "$STAGGER_SEC" -le 0 ]]; then
    return 0
  fi
  if [[ -n "$STAGGER_READY_RE" ]]; then
    echo "[stagger] wait ready re=/$STAGGER_READY_RE/ log=$log timeout=${STAGGER_READY_TIMEOUT}s ($label)"
    while true; do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "[stagger] WARN $label exited before ready (see $log)"
        return 1
      fi
      if grep -qE "$STAGGER_READY_RE" "$log" 2>/dev/null; then
        echo "[stagger] $label ready after $((SECONDS - t0))s"
        return 0
      fi
      if (( SECONDS - t0 >= STAGGER_READY_TIMEOUT )); then
        echo "[stagger] WARN $label ready-timeout ${STAGGER_READY_TIMEOUT}s, continue to play the next card"
        return 1
      fi
      sleep 5
    done
  fi
  echo "[stagger] sleep ${STAGGER_SEC}s before next shard ($label launched)"
  sleep "$STAGGER_SEC"
  return 0
}

launch_model() {
  local env="$1" mid="$2" script="$3"
  shift 3
  local extra_args=("$@")
  echo "########## $mid env=$env $(date '+%F %T') NGPU=$NGPU stagger_sec=$STAGGER_SEC ready_re=${STAGGER_READY_RE:-none} ##########"
  conda activate "$env"
  local pids=()
  local shard=0
  local phys
  for phys in "${GPU_ARR[@]}"; do
    local log="$LOGDIR/${mid}_shard${shard}.log"
    : >"$log"
    echo "[launch] $mid shard $shard/$NGPU -> GPU $phys  log=$log"
    (
      cd "$CODE"
      CUDA_VISIBLE_DEVICES="$phys" python -u "$script" \
        --pack "$PACK" \
        --shard-id "$shard" --num-shards "$NGPU" \
        "${LIMIT_ARGS[@]}" \
        "${SEED_ARGS[@]}" \
        "${extra_args[@]}"
    ) >"$log" 2>&1 &
    pids+=($!)
    # Off-peak: the last one shard no need to wait anymore
    if (( shard < NGPU - 1 )) && { [[ -n "$STAGGER_READY_RE" ]] || [[ "$STAGGER_SEC" -gt 0 ]]; }; then
      wait_shard_ready "$log" "${pids[-1]}" "$mid/shard$shard" || true
    fi
    shard=$((shard + 1))
  done
  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then fail=1; fi
  done
  local n_png out_mid="$mid"
  if [[ "$SEED_TAG" == "1" || "$SEED_TAG" == "true" || "$SEED_TAG" == "yes" ]] && [[ -n "$SEED" ]]; then
    out_mid="${mid}_s${SEED}"
  fi
  n_png=$(ls "$PACK/outputs/$out_mid"/*.png 2>/dev/null | wc -l || echo 0)
  echo "[done] $out_mid png=$n_png/$N fail=$fail"
  conda deactivate
  return $fail
}

# BAGEL: stagger starts + per-GPU --offload-folder (avoid 8 procs sharing /tmp/bagel_offload)
# MAX_MEM: default 20GiB on typical GPU hosts; high-VRAM hosts may use MAX_MEM=60GiB
launch_model_bagel() {
  local mid=bagel script=run_bagel.py
  local max_mem="${MAX_MEM:-20GiB}"
  echo "########## $mid env=bagel $(date '+%F %T') NGPU=$NGPU max_mem=$max_mem stagger_sec=$STAGGER_SEC ready_re=${STAGGER_READY_RE:-none} ##########"
  conda activate bagel
  local pids=()
  local shard=0
  local phys
  for phys in "${GPU_ARR[@]}"; do
    local log="$LOGDIR/${mid}_shard${shard}.log"
    local off="/tmp/bagel_offload_shard${shard}"
    mkdir -p "$off"
    : >"$log"
    echo "[launch] $mid shard $shard/$NGPU -> GPU $phys  offload=$off  log=$log"
    (
      cd "$CODE"
      CUDA_VISIBLE_DEVICES="$phys" python -u "$script" \
        --pack "$PACK" \
        --shard-id "$shard" --num-shards "$NGPU" \
        "${LIMIT_ARGS[@]}" \
        "${SEED_ARGS[@]}" \
        --max-mem "$max_mem" \
        --offload-folder "$off"
    ) >"$log" 2>&1 &
    pids+=($!)
    if (( shard < NGPU - 1 )) && { [[ -n "$STAGGER_READY_RE" ]] || [[ "$STAGGER_SEC" -gt 0 ]]; }; then
      wait_shard_ready "$log" "${pids[-1]}" "$mid/shard$shard" || true
    fi
    shard=$((shard + 1))
  done
  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then fail=1; fi
  done
  local n_png out_mid="$mid"
  if [[ "$SEED_TAG" == "1" || "$SEED_TAG" == "true" || "$SEED_TAG" == "yes" ]] && [[ -n "$SEED" ]]; then
    out_mid="${mid}_s${SEED}"
  fi
  n_png=$(ls "$PACK/outputs/$out_mid"/*.png 2>/dev/null | wc -l || echo 0)
  echo "[done] $out_mid png=$n_png/$N fail=$fail"
  conda deactivate
  return $fail
}

run_one() {
  case "$1" in
    kontext|flux|flux1-kontext-dev)
      if [[ "$HIGH_VRAM" == "1" ]]; then
        launch_model mpie_edit flux1-kontext-dev run_kontext.py --no-quant
      else
        launch_model mpie_edit flux1-kontext-dev run_kontext.py
      fi
      ;;
    qwen|qwen-image-edit-2511)
      if [[ "$HIGH_VRAM" == "1" ]]; then
        # Compatible with old runner(only --no-offload) and new runner（--offload none equivalence)
        launch_model mpie_edit qwen-image-edit-2511 run_qwen_edit.py --no-offload
      else
        launch_model mpie_edit qwen-image-edit-2511 run_qwen_edit.py --offload sequential
      fi
      ;;
    omnigen2)
      launch_model omnigen2 omnigen2 run_omnigen2.py --offload model
      ;;
    uno)
      launch_model uno uno run_uno.py --offload --model-type flux-dev-fp8
      ;;
    ace)
      launch_model ace ace run_ace.py --offload sequential --size 768
      ;;
    bagel)
      # BAGEL big weight+offload:ban 8 Roads are loaded simultaneously; each card is independent offload Table of contents
      if [[ -z "${STAGGER_READY_RE:-}" && "${STAGGER_SEC:-0}" -eq 0 ]]; then
        STAGGER_READY_RE='bagel ready'
      fi
      launch_model_bagel
      ;;
    dreamo)
      # DreamO int8: The default is to wait until the peak is shifted "dreamo ready" Play the next card again to avoid 8 simultaneous quantification of roads OOM
      if [[ -z "${STAGGER_READY_RE:-}" && "${STAGGER_SEC:-0}" -eq 0 ]]; then
        STAGGER_READY_RE='dreamo ready'
      fi
      launch_model dreamo dreamo run_dreamo.py --quant int8
      ;;
    firered|firered-image-edit|FireRed-Image-Edit-1.1)
      if [[ -z "${STAGGER_READY_RE:-}" && "${STAGGER_SEC:-0}" -eq 0 ]]; then
        STAGGER_READY_RE='firered ready'
      fi
      # high-VRAM GPU: keep full model on device; weight dir name FireRed-Image-Edit-1.1
      launch_model firered firered run_firered.py --offload none --max-refs 3
      ;;
    *)
      echo "Unknown model: $1"
      return 2
      ;;
  esac
}

WHAT="${1:-all}"
rc=0
MODELS=()
case "$WHAT" in
  all|both)
    MODELS=(kontext qwen omnigen2 uno ace bagel dreamo)
    ;;
  *)
    MODELS=("$WHAT")
    ;;
esac

for m in "${MODELS[@]}"; do
  run_one "$m" || rc=1
done

echo "---- coverage ----"
# When --seed-tag is on, outputs live under <base>_s<seed>/ (not bare <base>/).
_cov_suffix=""
if [[ "$SEED_TAG" == "1" || "$SEED_TAG" == "true" || "$SEED_TAG" == "yes" ]] && [[ -n "$SEED" ]]; then
  _cov_suffix="_s${SEED}"
  echo "  (seed-tag: counting outputs/*${_cov_suffix}/)"
fi
for mid in flux1-kontext-dev qwen-image-edit-2511 omnigen2 uno ace bagel dreamo firered; do
  n=$(ls "$PACK/outputs/${mid}${_cov_suffix}"/*.png 2>/dev/null | wc -l || echo 0)
  echo "  ${mid}${_cov_suffix} png=$n / $N"
done
echo "logs: $LOGDIR"
exit $rc
