#!/bin/bash
# MPIE pipeline smoke test — run the full chain with mpie_testkit before real data lands
# Prereq: conda env mpie ready (ALL READY) + ~/mpie_testkit unpacked
# Usage: cd $MPIE_ROOT/code && bash smoke_test.sh 2>&1 | tee /tmp/mpie_smoke.log
set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mpie
export MPIE_WEIGHTS_DIR=${MPIE_WEIGHTS_DIR:-$HOME/mpie_weights/ultralytics}

T=~/mpie_smoke_run          # disposable test workspace
DB=$T/manifests/test.db
rm -rf $T && mkdir -p $T/manifests

echo "===== Stage 2: shot split + 4fps extract ====="
python pipeline/02_extract_frames/shot_split_extract.py \
  --videos ~/mpie_testkit/videos --dataset testkit --license cc0 \
  --out $T --db $DB --fps 4

echo "===== Stage 3: coarse scan (person count + IoU + motion) ====="
python pipeline/03_coarse_scan/coarse_scan.py --frames $T/frames --db $DB

echo "===== Stage 4: interaction density refine + peak pick ====="
python pipeline/04_peak_refine/interaction_density.py --db $DB --top-per-video 10

echo "===== Stage 5: person crop + identity cluster ====="
python pipeline/05_track_identity/identity_cluster.py --db $DB --out $T/crops
python pipeline/05_track_identity/identity_cluster.py --db $DB --out $T/crops --cluster

echo "===== Stage 6: cross-frame reference crops ====="
python pipeline/06_ref_crop/ref_crop.py --db $DB --out $T/crops

echo "===== eval metrics (pilot generated images) ====="
python eval/metrics_lite.py \
  --img ~/mpie_testkit/pilot_images/01_judo_throw__seedream-5.png \
  --refs ~/mpie_testkit/pilot_images/_refs/5b524abc_120030323.jpg \
         ~/mpie_testkit/pilot_images/_refs/51383bc5_120022399_1.jpg \
  --n 2

echo "===== DB row counts ====="
python - <<'PY'
import sqlite3, os
db = os.path.expanduser("~/mpie_smoke_run/manifests/test.db")
c = sqlite3.connect(db)
for t in ("videos","shots","keyframes","persons","refs"):
    n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    sel = ""
    if t == "keyframes":
        sel = f" (selected={c.execute('SELECT COUNT(*) FROM keyframes WHERE selected=1').fetchone()[0]})"
    print(f"{t:10s}: {n}{sel}")
PY
echo "===== SMOKE TEST done: spot-check ~/mpie_smoke_run/crops/{persons,refs_raw,refs_clean} ====="
