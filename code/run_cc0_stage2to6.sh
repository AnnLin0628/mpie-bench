#!/bin/bash
# CC0 batch1 Stage 2→6 on a GPU host — first full run on real data
# Persists to ~/mpie_data (not smoke temp); idempotent via DB upsert
# Usage: bash run_cc0_stage2to6.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mpie
export MPIE_WEIGHTS_DIR=${MPIE_WEIGHTS_DIR:-$HOME/mpie_weights/ultralytics}
cd $MPIE_ROOT/code

VID=~/mpie_data/raw_video/cc0_pexels
DATA=~/mpie_data
DB=$DATA/manifests/mpie.db
mkdir -p "$DATA/manifests"

echo "===== video count ====="
find "$VID" -name "*.mp4" | wc -l

echo "===== Stage 2: shot split + 4fps extract ====="
python pipeline/02_extract_frames/shot_split_extract.py \
  --videos "$VID" --dataset cc0_pexels --license cc0 \
  --out "$DATA" --db "$DB" --fps 4

echo "===== Stage 3: coarse scan ====="
python pipeline/03_coarse_scan/coarse_scan.py --frames "$DATA/frames" --db "$DB"

echo "===== Stage 4: interaction density refine + peak pick ====="
python pipeline/04_peak_refine/interaction_density.py --db "$DB" --top-per-video 8

echo "===== Stage 5: person crop + identity cluster ====="
python pipeline/05_track_identity/identity_cluster.py --db "$DB" --out "$DATA/crops"
python pipeline/05_track_identity/identity_cluster.py --db "$DB" --out "$DATA/crops" --cluster

echo "===== Stage 6: cross-frame reference crops ====="
python pipeline/06_ref_crop/ref_crop.py --db "$DB" --out "$DATA/crops"

echo "===== DB stats ====="
python - <<'PY'
import sqlite3, os
db = os.path.expanduser("~/mpie_data/manifests/mpie.db")
c = sqlite3.connect(db)
for t in ("videos","shots","keyframes","persons","refs"):
    n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    extra = ""
    if t == "keyframes":
        sel = c.execute("SELECT COUNT(*) FROM keyframes WHERE selected=1").fetchone()[0]
        extra = f" (selected={sel})"
    if t == "persons":
        idn = c.execute("SELECT COUNT(DISTINCT identity_id) FROM persons").fetchone()[0]
        extra = f" ({idn} identities)"
    print(f"{t:10s}: {n}{extra}")
# selected keyframes per interaction type (path prefix)
print("--- selected keyframes per type (balance spot-check) ---")
rows = c.execute("""SELECT v.path, COUNT(*) FROM keyframes k
    JOIN videos v ON v.video_id=k.video_id WHERE k.selected=1
    GROUP BY v.path""").fetchall()
from collections import Counter
cnt = Counter()
for p, n in rows:
    typ = p.split("/cc0_pexels/")[-1].split("/")[0] if "/cc0_pexels/" in p else "?"
    cnt[typ] += n
for typ, n in cnt.most_common():
    print(f"  {typ:22s} {n}")
PY
echo "===== done: spot-check ~/mpie_data/crops/{persons,refs_clean} ====="
echo "Share the DB stats above with the team"
