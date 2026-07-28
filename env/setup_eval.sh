#!/usr/bin/env bash
# Public helper for evaluation weights / sanity checks (machine-agnostic).
# Usage:
#   bash env/setup_eval.sh --arcface-weights
#   bash env/setup_eval.sh --check-hps
#   bash env/setup_eval.sh --check-arcface
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MPIE_WEIGHTS="${MPIE_WEIGHTS:-$HOME/mpie_weights}"
HPS_DIR="${MPIE_HPS_WEIGHTS:-$MPIE_WEIGHTS/hpsv2}"

arcface_weights() {
  python - <<'PY'
from pathlib import Path
import io, urllib.request, zipfile
root = Path.home() / ".insightface" / "models" / "antelopev2"
need = ["1k3d68.onnx", "2d106det.onnx", "genderage.onnx", "glintr100.onnx", "scrfd_10g_bnkps.onnx"]
if root.is_dir() and all((root / n).exists() for n in need):
    print("antelopev2 already present:", root)
    raise SystemExit(0)
root.parent.mkdir(parents=True, exist_ok=True)
url = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
print("downloading", url)
data = urllib.request.urlopen(url, timeout=180).read()
with zipfile.ZipFile(io.BytesIO(data)) as zf:
    zf.extractall(root.parent)
print("extracted under", root.parent)
PY
}

check_hps() {
  mkdir -p "$HPS_DIR"
  local ok=1
  for f in open_clip_pytorch_model.bin; do
    if [[ ! -f "$HPS_DIR/$f" ]]; then
      echo "MISSING $HPS_DIR/$f"
      ok=0
    else
      ls -lh "$HPS_DIR/$f"
    fi
  done
  if [[ -f "$HPS_DIR/HPS_v2.1_compressed.pt" ]]; then
    ls -lh "$HPS_DIR/HPS_v2.1_compressed.pt"
  elif [[ -f "$HPS_DIR/HPS_v2_compressed.pt" ]]; then
    echo "note: using HPS_v2_compressed.pt (v2.0 fallback)"
    ls -lh "$HPS_DIR/HPS_v2_compressed.pt"
  else
    echo "MISSING $HPS_DIR/HPS_v2.1_compressed.pt (or HPS_v2_compressed.pt)"
    ok=0
  fi
  if [[ "$ok" -ne 1 ]]; then
    echo "Place HPSv2 weights under $HPS_DIR (see docs/INSTALL.md)."
    exit 1
  fi
  echo "HPSv2 weights OK under $HPS_DIR"
}

check_arcface() {
  python - <<'PY'
import numpy as np
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
_ = app.get(np.zeros((256, 256, 3), dtype=np.uint8))
print("insightface antelopev2 ready")
PY
}

usage() {
  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
}

cmd="${1:-}"
case "$cmd" in
  --arcface-weights) arcface_weights ;;
  --check-hps) check_hps ;;
  --check-arcface) check_arcface ;;
  -h|--help|"") usage; exit 0 ;;
  *) echo "Unknown: $cmd"; usage; exit 1 ;;
esac
