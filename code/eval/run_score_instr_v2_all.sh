#!/usr/bin/env bash
# Score Instr v2 for all models that already have outputs/ under the pack.
set -euo pipefail

PACK="${PACK:-$HOME/mpie_testset_pack}"
WORKERS="${WORKERS:-8}"
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -d "${PACK}/outputs" ]]; then
  echo "ERROR: no outputs/ under $PACK"
  exit 1
fi

for d in "${PACK}/outputs"/*/; do
  [[ -d "$d" ]] || continue
  mid="$(basename "$d")"
  # skip empty
  n=$(find "$d" -maxdepth 1 \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | wc -l)
  if [[ "$n" -lt 1 ]]; then
    echo "skip $mid (no images)"
    continue
  fi
  echo "=== scoring Instr v2: $mid (n_img~$n) ==="
  PACK="$PACK" bash "$EVAL_DIR/run_score_instr_v2.sh" "$mid" "$WORKERS"
  sleep 1
done

echo "all launched; use report_instr_v2.py when done"
