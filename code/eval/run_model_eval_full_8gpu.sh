#!/usr/bin/env bash
# multi-GPU: Single model full quantity 2500 → ID + Qual + Mesh(per stage 8 card sharding)
#
# usage:
#   export MPIE_TEST_PACK=~/mpie_testset_pack
#   bash run_model_eval_full_8gpu.sh flux1-kontext-dev
#   bash run_model_eval_full_8gpu.sh dreamo          # Also possible
#   NGPU=4 bash run_model_eval_full_8gpu.sh flux1-kontext-dev
#
# log:/tmp/mpie_eval_<model>/
set -euo pipefail

MID="${1:-${MODEL:-}}"
if [[ -z "$MID" ]]; then
  echo "usage: $0 <model_id>   e.g.  $0 flux1-kontext-dev"
  exit 1
fi

PACK="${MPIE_TEST_PACK:-$HOME/mpie_testset_pack}"
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
NGPU="${NGPU:-8}"
LOGDIR="${LOGDIR:-/tmp/mpie_eval_${MID}}"
MULTIHMR_REPO="${MULTIHMR_REPO:-$HOME/models/multi-hmr}"
mkdir -p "$LOGDIR"

source "$(conda info --base)/etc/profile.d/conda.sh"

n_png=$(ls "$PACK/outputs/$MID"/*.png 2>/dev/null | wc -l || echo 0)
echo "==== $MID  pack=$PACK  NGPU=$NGPU  png=$n_png ===="
if [[ "$n_png" -lt 100 ]]; then
  echo "ERROR: too few png under $PACK/outputs/$MID"
  exit 1
fi

count_json() {
  local d="$1"
  find "$d" -maxdepth 1 -name '*.json' ! -name '_*' 2>/dev/null | wc -l
}

# ---------- 1) ID ----------
echo "==== [1/3] ArcFace ID ===="
conda activate "${MPIE_EVAL_ENV:-mpie}"
cd "$EVAL_DIR"
pids=()
for g in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$g python -u score_arcface_v1.py \
    --pack "$PACK" --model-id "$MID" --ctx-id 0 \
    --shard-id "$g" --num-shards "$NGPU" \
    >"$LOGDIR/arcface_s${g}.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}" || true
n_id=$(count_json "$PACK/judgments/arcface_v1/$MID")
echo "[ID] done  n=$n_id  logs=$LOGDIR/arcface_s*.log"

# ---------- 2) Qual ----------
echo "==== [2/3] HPSv2 Qual ===="
pids=()
for g in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$g python -u score_hpsv2.py \
    --pack "$PACK" --model-id "$MID" \
    --shard-id "$g" --num-shards "$NGPU" \
    >"$LOGDIR/hps_s${g}.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}" || true
n_q=$(count_json "$PACK/judgments/hpsv2/$MID")
echo "[Qual] done  n=$n_q  logs=$LOGDIR/hps_s*.log"

# ---------- 3) Mesh Anat+Inter ----------
echo "==== [3/3] Mesh Anat+Inter ===="
conda activate multihmr
export PYTHONPATH="$MULTIHMR_REPO:${PYTHONPATH:-}"
cd "$EVAL_DIR"
if [[ ! -f "$PACK/judgments/mesh_v3/_calibration.json" ]]; then
  echo "[mesh] calibrating from GT (limit 80)..."
  CUDA_VISIBLE_DEVICES=0 python -u score_mesh_v3.py \
    --pack "$PACK" --multihmr-repo "$MULTIHMR_REPO" \
    --gt-only --calibrate --limit 80 \
    >"$LOGDIR/mesh_gt_calib.log" 2>&1 || echo "WARN: calib failed, see $LOGDIR/mesh_gt_calib.log"
fi
pids=()
for g in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$g python -u score_mesh_v3.py \
    --pack "$PACK" --multihmr-repo "$MULTIHMR_REPO" \
    --model-id "$MID" \
    --shard-id "$g" --num-shards "$NGPU" \
    >"$LOGDIR/mesh_s${g}.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}" || true

python rescore_mesh_inter.py "$PACK/judgments/mesh_v3" --pack "$PACK" \
  >"$LOGDIR/mesh_rescore_inter.log" 2>&1 || true
if [[ -f rescore_mesh_anat.py ]]; then
  python rescore_mesh_anat.py "$PACK/judgments/mesh_v3" \
    >"$LOGDIR/mesh_rescore_anat.log" 2>&1 || true
fi

# merge mesh summary
python - <<PY
import json
from pathlib import Path
try:
    from rescore_mesh_inter import summarize
except Exception:
    summarize = None
root = Path("$PACK") / "judgments" / "mesh_v3" / "$MID"
recs = []
for p in sorted(root.glob("*.json")):
    if p.name.startswith("_"):
        continue
    recs.append(json.loads(p.read_text()))
if recs and summarize:
    s = summarize(recs)
    (root / "_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2))
    print("[mesh] summary", {k: s.get(k) for k in ("n", "n_ok", "S_anat_mesh", "S_inter_mesh")})
else:
    print("[mesh] n_json=", len(recs))
PY

n_m=$(count_json "$PACK/judgments/mesh_v3/$MID")
echo "==== DONE $MID ===="
echo "  png=$n_png  ID=$n_id  Qual=$n_q  Mesh=$n_m"
echo "  logs: $LOGDIR"
echo "  pack: tar czf ~/${MID}_judgments_all_\$(date +%Y%m%d).tgz \\"
echo "    judgments/arcface_v1/$MID judgments/hpsv2/$MID \\"
echo "    judgments/mesh_v3/$MID judgments/mesh_v3/_calibration.json"
