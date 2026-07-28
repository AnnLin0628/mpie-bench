#!/usr/bin/env bash
# multi-GPU:  smoke100 full model mesh v3(Already have judgments will be skipped unless FORCE=1）
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multihmr

export PACK="${PACK:-$HOME/mpie_testset_pack}"
export MULTIHMR_REPO="${MULTIHMR_REPO:-$HOME/models/multi-hmr}"
export PYTHONPATH="$MULTIHMR_REPO:${PYTHONPATH:-}"
export MPIE_BENCH="${MPIE_BENCH:-$HOME/mpie_bench}"
FORCE_FLAG="${FORCE:-}"   # FORCE=1 Forced rerun

cd "$MPIE_BENCH/code/eval"
test -f score_mesh_v3.py
test -f "$PACK/manifest.jsonl"

models=()
if [[ -d "$PACK/outputs" ]]; then
  while IFS= read -r d; do models+=("$(basename "$d")"); done < <(find "$PACK/outputs" -mindepth 1 -maxdepth 1 -type d | sort)
fi
echo "models (${#models[@]}): ${models[*]:-none}"

extra=()
[[ "$FORCE_FLAG" == "1" ]] && extra+=(--force)

# GT Already available and can be skipped; timing needs to be recalibrated:GT_CALIB=1
if [[ "${GT_CALIB:-0}" == "1" ]]; then
  python score_mesh_v3.py --pack "$PACK" --multihmr-repo "$MULTIHMR_REPO" \
    --gt-only --calibrate "${extra[@]}"
fi

for mid in "${models[@]}"; do
  echo "===== mesh $mid ====="
  python score_mesh_v3.py --pack "$PACK" --multihmr-repo "$MULTIHMR_REPO" \
    --model-id "$mid" "${extra[@]}"
done

echo "===== rescore all (prompt penalty Inter + residual-heavy Anat) ====="
python rescore_mesh_inter.py "$PACK/judgments/mesh_v3" "$PACK"
python rescore_mesh_anat.py "$PACK/judgments/mesh_v3"

echo "===== summaries ====="
for s in "$PACK"/judgments/mesh_v3/*/\_summary.json; do
  echo "--- $s ---"
  python -c "import json,sys; d=json.load(open(sys.argv[1])); print({k:d.get(k) for k in ['n','n_ok','S_anat_mesh','S_anat_overcount_mean','P_anat_extra_mean','anat_leftover_frac_mean','anat_n_leftover_blobs_mean','S_inter_mesh','recon_fail_rate']})" "$s"
done

echo "done. pack judgments:"
echo "  cd $PACK && tar czf ~/smoke100_judgments_mesh_v3_\$(date +%Y%m%d).tgz judgments/mesh_v3"
