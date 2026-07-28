#!/usr/bin/env bash
# MPIE-Bench end-to-end evaluation (protocol v3, six axes).
#
# After generations exist under $MPIE_TEST_PACK/outputs/<model_id>/, this script
# scores Count · Identity · Anatomy · Interaction · Instruction · Quality and
# writes a leaderboard aggregate.
#
# Usage:
#   cp configs/eval.env.example configs/eval.env   # edit paths + API
#   bash code/eval/run_eval_e2e.sh --model flux1-kontext-dev
#   bash code/eval/run_eval_e2e.sh --model my-model --axes id,qual,mesh
#   bash code/eval/run_eval_e2e.sh --all-models
#   bash code/eval/run_eval_e2e.sh --model my-model --aggregate-only
#
# Axes (comma-separated via --axes):
#   count | instr | id | qual | mesh | aggregate | all
#   mesh = Anatomy + Interaction (Multi-HMR)
#
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$EVAL_DIR/../.." && pwd)"
cd "$EVAL_DIR"

# ---------- load config ----------
_load_env() {
  local f
  for f in \
    "${MPIE_EVAL_ENV_FILE:-}" \
    "$REPO_ROOT/configs/eval.env" \
    "${HOME}/.mpie_env"
  do
    [[ -n "$f" && -f "$f" ]] || continue
    set -a
    # shellcheck disable=SC1090
    source "$f"
    set +a
    echo "[env] loaded $f"
  done
}
_load_env

PACK="${MPIE_TEST_PACK:-${PACK:-$REPO_ROOT/data/testset}}"
MULTIHMR_REPO="${MULTIHMR_REPO:-$HOME/models/multi-hmr}"
NGPU="${NGPU:-1}"
VLM_WORKERS="${VLM_WORKERS:-8}"
JUDGE="${MPIE_JUDGE_MODEL:-${JUDGE:-gpt-5.5}}"
LIMIT="${EVAL_LIMIT:-0}"
OUT_DIR="${MPIE_EVAL_OUT:-$REPO_ROOT/data/eval_outputs/latest}"
LOGROOT="${LOGROOT:-/tmp/mpie_eval_e2e}"
MPIE_EVAL_ENV="${MPIE_EVAL_ENV:-mpie}"
MPIE_MESH_ENV="${MPIE_MESH_ENV:-multihmr}"

export AI_GATEWAY_KEY="${AI_GATEWAY_KEY:-${MPIE_VLM_KEY:-${AI_GATEWAY_KEY:-}}}"
export AI_GATEWAY_URL="${AI_GATEWAY_URL:-${AI_GATEWAY_URL:-}}"
export MPIE_TEST_PACK="$PACK"

MODEL=""
ALL_MODELS=0
AGGREGATE_ONLY=0
AXES="all"
SKIP_INSTR_BUILD=0
DRY_RUN=0

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|-m) MODEL="${2:-}"; shift 2 ;;
    --all-models) ALL_MODELS=1; shift ;;
    --axes) AXES="${2:-all}"; shift 2 ;;
    --aggregate-only) AGGREGATE_ONLY=1; AXES="aggregate"; shift ;;
    --pack) PACK="${2:-}"; export MPIE_TEST_PACK="$PACK"; shift 2 ;;
    --out) OUT_DIR="${2:-}"; shift 2 ;;
    --ngpu) NGPU="${2:-}"; shift 2 ;;
    --limit) LIMIT="${2:-}"; shift 2 ;;
    --judge) JUDGE="${2:-}"; shift 2 ;;
    --skip-instr-build) SKIP_INSTR_BUILD=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown arg: $1"; usage 1 ;;
  esac
done

# ---------- helpers ----------
have_axis() {
  local want="$1"
  [[ "$AXES" == "all" ]] && return 0
  [[ ",${AXES}," == *",${want},"* ]] && return 0
  return 1
}

count_json() {
  local d="$1"
  find "$d" -maxdepth 1 -name '*.json' ! -name '_*' 2>/dev/null | wc -l | tr -d ' '
}

list_models() {
  if [[ "$ALL_MODELS" -eq 1 ]]; then
    find "$PACK/outputs" -mindepth 1 -maxdepth 1 -type d ! -name '_*' -printf '%f\n' 2>/dev/null | sort
  elif [[ -n "$MODEL" ]]; then
    echo "$MODEL"
  else
    echo "ERROR: pass --model <id> or --all-models" >&2
    exit 1
  fi
}

activate_eval_py() {
  if [[ -n "${PYTHON_MPIE:-}" ]]; then
    PY="$PYTHON_MPIE"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$MPIE_EVAL_ENV"
  fi
  PY="${PY:-python}"
}

activate_mesh_py() {
  if [[ -n "${PYTHON_MESH:-}" ]]; then
    PY="$PYTHON_MESH"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$MPIE_MESH_ENV"
  fi
  PY="${PY:-python}"
  export PYTHONPATH="${MULTIHMR_REPO}:${PYTHONPATH:-}"
}

run_gpu_shards() {
  # run_gpu_shards <log_prefix> <python_module_args...>
  # Uses CUDA_VISIBLE_DEVICES=$g and --shard-id / --num-shards
  local prefix="$1"; shift
  local g pids=()
  if [[ "$NGPU" -le 1 ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" -u "$@" \
      --shard-id 0 --num-shards 1 \
      >"${LOGDIR}/${prefix}.log" 2>&1
    return $?
  fi
  for g in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$g "$PY" -u "$@" \
      --shard-id "$g" --num-shards "$NGPU" \
      >"${LOGDIR}/${prefix}_s${g}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  return "$rc"
}

limit_args() {
  if [[ "${LIMIT}" -gt 0 ]]; then
    echo --limit "$LIMIT"
  fi
}

preflight() {
  echo "==== MPIE-Bench E2E eval ===="
  echo "  pack=$PACK"
  echo "  axes=$AXES"
  echo "  ngpu=$NGPU  vlm_workers=$VLM_WORKERS  judge=$JUDGE  limit=$LIMIT"
  echo "  out=$OUT_DIR"
  echo "  multihmr=$MULTIHMR_REPO"
  [[ -d "$PACK" ]] || { echo "ERROR: pack not found: $PACK"; exit 1; }
  [[ -d "$PACK/outputs" ]] || { echo "ERROR: missing $PACK/outputs"; exit 1; }
  if have_axis count || have_axis instr; then
    if [[ -z "${AI_GATEWAY_KEY:-}" && -z "${AI_GATEWAY_KEY:-}" ]]; then
      echo "WARN: no AI_GATEWAY_KEY set; Count/Instr API calls will fail."
    fi
  fi
  if have_axis mesh; then
    [[ -d "$MULTIHMR_REPO" ]] || echo "WARN: MULTIHMR_REPO missing: $MULTIHMR_REPO"
  fi
}

score_count() {
  local mid="$1"
  echo "==== [Count] VLM judge → $mid ===="
  activate_eval_py
  # shellcheck disable=SC2046
  "$PY" -u score_vlm_v1.py \
    --pack "$PACK" --model-id "$mid" --judge-model "$JUDGE" \
    --workers "$VLM_WORKERS" --resume \
    $(limit_args) \
    >"$LOGDIR/count.log" 2>&1 || {
      echo "ERROR: Count failed; see $LOGDIR/count.log"; return 1; }
  echo "[Count] n=$(count_json "$PACK/judgments/vlm_judge_v1/$mid")"
}

ensure_instr_bank() {
  local n
  n=$(count_json "$PACK/instr_qa_v2")
  if [[ "$n" -gt 0 ]]; then
    echo "[Instr] frozen QA bank present (n=$n)"
    return 0
  fi
  if [[ "$SKIP_INSTR_BUILD" -eq 1 ]]; then
    echo "ERROR: missing $PACK/instr_qa_v2/ and --skip-instr-build set"
    return 1
  fi
  echo "==== [Instr] building frozen QA bank (one-time) ===="
  activate_eval_py
  # shellcheck disable=SC2046
  "$PY" -u build_instr_qa_v2.py \
    --pack "$PACK" --qg-model "$JUDGE" --workers "$VLM_WORKERS" \
    --filter --resume \
    $(limit_args) \
    >"$LOGDIR/instr_build.log" 2>&1 || {
      echo "ERROR: Instr QA build failed; see $LOGDIR/instr_build.log"; return 1; }
  echo "[Instr] bank n=$(count_json "$PACK/instr_qa_v2")"
}

score_instr() {
  local mid="$1"
  ensure_instr_bank || return 1
  echo "==== [Instr] Instr v2 → $mid ===="
  activate_eval_py
  # shellcheck disable=SC2046
  "$PY" -u score_instr_v2.py \
    --pack "$PACK" --model-id "$mid" --judge-model "$JUDGE" \
    --workers "$VLM_WORKERS" --resume \
    $(limit_args) \
    >"$LOGDIR/instr.log" 2>&1 || {
      echo "ERROR: Instr failed; see $LOGDIR/instr.log"; return 1; }
  echo "[Instr] n=$(count_json "$PACK/judgments/instr_v2/$mid")"
}

score_id() {
  local mid="$1"
  echo "==== [ID] ArcFace → $mid ===="
  activate_eval_py
  # shellcheck disable=SC2046
  run_gpu_shards arcface score_arcface_v1.py \
    --pack "$PACK" --model-id "$mid" --ctx-id 0 \
    $(limit_args) || {
      echo "ERROR: ID failed; see $LOGDIR/arcface*.log"; return 1; }
  echo "[ID] n=$(count_json "$PACK/judgments/arcface_v1/$mid")"
}

score_qual() {
  local mid="$1"
  echo "==== [Qual] HPSv2 → $mid ===="
  activate_eval_py
  # shellcheck disable=SC2046
  run_gpu_shards hps score_hpsv2.py \
    --pack "$PACK" --model-id "$mid" \
    $(limit_args) || {
      echo "ERROR: Qual failed; see $LOGDIR/hps*.log"; return 1; }
  echo "[Qual] n=$(count_json "$PACK/judgments/hpsv2/$mid")"
}

score_mesh() {
  local mid="$1"
  echo "==== [Anat+Inter] Multi-HMR mesh → $mid ===="
  activate_mesh_py
  if [[ ! -f "$PACK/judgments/mesh_v3/_calibration.json" ]]; then
    echo "[mesh] calibrating from GT (limit 80)..."
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" -u score_mesh_v3.py \
      --pack "$PACK" --multihmr-repo "$MULTIHMR_REPO" \
      --gt-only --calibrate --limit 80 \
      >"$LOGDIR/mesh_gt_calib.log" 2>&1 || \
      echo "WARN: calib failed; see $LOGDIR/mesh_gt_calib.log"
  fi
  # shellcheck disable=SC2046
  run_gpu_shards mesh score_mesh_v3.py \
    --pack "$PACK" --multihmr-repo "$MULTIHMR_REPO" --model-id "$mid" \
    $(limit_args) || {
      echo "ERROR: Mesh failed; see $LOGDIR/mesh*.log"; return 1; }

  "$PY" -u rescore_mesh_inter.py "$PACK/judgments/mesh_v3" --pack "$PACK" \
    >"$LOGDIR/mesh_rescore_inter.log" 2>&1 || true
  if [[ -f rescore_mesh_anat.py ]]; then
    "$PY" -u rescore_mesh_anat.py "$PACK/judgments/mesh_v3" \
      >"$LOGDIR/mesh_rescore_anat.log" 2>&1 || true
  fi
  if [[ -f resummarize_mesh_v3.py ]]; then
    "$PY" -u resummarize_mesh_v3.py "$PACK/judgments/mesh_v3" \
      >"$LOGDIR/mesh_resummarize.log" 2>&1 || true
  fi
  echo "[Mesh] n=$(count_json "$PACK/judgments/mesh_v3/$mid")"
}

run_aggregate() {
  echo "==== [Aggregate] six-axis leaderboard ===="
  activate_eval_py
  mkdir -p "$OUT_DIR"
  local models_args=()
  if [[ "$ALL_MODELS" -eq 0 && -n "$MODEL" ]]; then
    models_args=(--models "$MODEL")
  fi
  "$PY" -u aggregate_mesh_v3.py \
    --pack "$PACK" --out "$OUT_DIR" "${models_args[@]}" \
    >"$LOGDIR/aggregate.log" 2>&1 || {
      echo "ERROR: aggregate failed; see $LOGDIR/aggregate.log"; return 1; }
  echo "[Aggregate] wrote $OUT_DIR/index.html and summary.json"
  if [[ -f "$OUT_DIR/summary.json" ]]; then
    SUMMARY_JSON="$OUT_DIR/summary.json" "$PY" -u <<'PY' || true
import json, os
from pathlib import Path
d = json.loads(Path(os.environ["SUMMARY_JSON"]).read_text())
by = d.get("by_model") or {}
order = d.get("ranking") or list(by.keys())
print("--- scores (protocol v3) ---")
print(f"{'model':28} {'Count':>6} {'ID':>6} {'Anat':>6} {'Inter':>6} {'Instr':>6} {'Qual':>6}")

def fmt(disp, m, *keys):
    for key in keys:
        v = disp.get(key, m.get(key))
        if v is None:
            continue
        try:
            return f"{float(v):.3f}"
        except Exception:
            return str(v)[:6]
    return "  — "

for mid in order:
    m = by.get(mid) or {}
    disp = m.get("display") or m
    print(
        f"{mid:28} "
        f"{fmt(disp, m, 'Count', 'S_count'):>6} "
        f"{fmt(disp, m, 'ID', 'S_id'):>6} "
        f"{fmt(disp, m, 'Anat', 'S_anat_mesh'):>6} "
        f"{fmt(disp, m, 'Inter', 'S_inter_mesh'):>6} "
        f"{fmt(disp, m, 'Instr', 'S_instr'):>6} "
        f"{fmt(disp, m, 'Qual', 'HPSv2'):>6}"
    )
PY
  fi
}

eval_one_model() {
  local mid="$1"
  LOGDIR="$LOGROOT/${mid}"
  mkdir -p "$LOGDIR"
  local n_png
  n_png=$(ls "$PACK/outputs/$mid"/*.png 2>/dev/null | wc -l | tr -d ' ')
  echo ""
  echo "######## model=$mid  png=$n_png  logs=$LOGDIR ########"
  if [[ "$n_png" -lt 1 && "$AGGREGATE_ONLY" -eq 0 ]]; then
    echo "ERROR: no png under $PACK/outputs/$mid"
    return 1
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] would score axes=$AXES"
    return 0
  fi
  have_axis count && score_count "$mid"
  have_axis instr && score_instr "$mid"
  have_axis id && score_id "$mid"
  have_axis qual && score_qual "$mid"
  have_axis mesh && score_mesh "$mid"
  echo "==== done model=$mid ===="
}

# ---------- main ----------
preflight
mapfile -t MODELS < <(list_models)
[[ "${#MODELS[@]}" -ge 1 ]] || { echo "ERROR: no models to score"; exit 1; }

if [[ "$AGGREGATE_ONLY" -eq 0 ]]; then
  for mid in "${MODELS[@]}"; do
    eval_one_model "$mid"
  done
fi

if have_axis aggregate; then
  LOGDIR="$LOGROOT/_aggregate"
  mkdir -p "$LOGDIR"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] would aggregate → $OUT_DIR"
  else
    run_aggregate
  fi
fi

echo ""
echo "All requested stages finished."
echo "  judgments: $PACK/judgments/"
echo "  report:    $OUT_DIR/index.html"
echo "  logs:      $LOGROOT/"
