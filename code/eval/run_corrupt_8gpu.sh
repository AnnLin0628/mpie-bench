#!/usr/bin/env bash
# 8-GPU parallel corruption (limit=200 by default).
# Usage on a GPU host:
#   export MPIE_ROOT=$PWD PACK=$PWD/data/testset
#   export MULTIHMR_REPO=/path/to/multi-hmr PY=python
#   bash $MPIE_ROOT/code/eval/run_corrupt_8gpu.sh
set -euo pipefail

MPIE_ROOT="${MPIE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PACK="${PACK:-$MPIE_ROOT/data/testset}"
MULTIHMR_REPO="${MULTIHMR_REPO:-$HOME/models/multi-hmr}"
PY="${PY:-python}"
LIMIT="${LIMIT:-200}"
# physical GPU ids (override if some busy): GPUS="0 1 2 3 4 5 6 7"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}

EVAL="$MPIE_ROOT/code/eval"
REVIEW="${MPIE_ANALYSIS_OUT:-$MPIE_ROOT/analysis/out}"
SHARD_DIR="$REVIEW/corrupt_shards_n${LIMIT}"
mkdir -p "$SHARD_DIR" "$REVIEW/logs"

echo "[8gpu] LIMIT=$LIMIT  GPUS=(${GPU_ARR[*]})  N=$N"
echo "[8gpu] MULTIHMR_REPO=$MULTIHMR_REPO"

pids=()
for i in "${!GPU_ARR[@]}"; do
  gid="${GPU_ARR[$i]}"
  out="$SHARD_DIR/shard${i}.json"
  log="$REVIEW/logs/corrupt_n${LIMIT}_shard${i}_gpu${gid}.log"
  echo "[8gpu] launch shard $i on GPU $gid → $out"
  CUDA_VISIBLE_DEVICES="$gid" "$PY" "$EVAL/corrupt_mesh_on_recon.py" \
    --pack "$PACK" \
    --multihmr-repo "$MULTIHMR_REPO" \
    --source gt \
    --limit "$LIMIT" \
    --num-shards "$N" \
    --shard-id "$i" \
    --out "$out" \
    >"$log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "[8gpu] pid $pid failed" >&2
    fail=1
  fi
done
if [[ "$fail" != "0" ]]; then
  echo "[8gpu] some shards failed; check $REVIEW/logs/" >&2
  exit 1
fi

MERGED="$REVIEW/analysis_corruption_recon_n${LIMIT}.json"
echo "[8gpu] merging → $MERGED"
"$PY" "$EVAL/corrupt_mesh_on_recon.py" \
  --merge-shards "$SHARD_DIR"/shard*.json \
  --out "$MERGED"

echo "=== DONE ==="
echo "Merged: $MERGED"
"$PY" - <<PY
import json
from pathlib import Path
d=json.loads(Path("$MERGED").read_text())
print("n_ok", d["n_ok"], "/", d["n_total_attempted"])
print("monotonic_ok", d["monotonic_ok"])
for k,v in d["delta_vs_baseline"].items():
    print(f"  {k}: ΔInter={v['delta_Inter']:+.3f} n={v['n']}")
PY
