#!/bin/bash
# Harmony4D Decompress package by package + Structural Inventory (on a GPU host, After the download is complete)
# one by one zip Delete by default after decompression zip Explosion-proof disk(327G×2); Want to keep it? KEEP_ZIP=1
# usage: bash harmony4d_unpack.sh ~/mpie_data/raw_video/harmony4d
set -e
ROOT=${1:-$HOME/mpie_data/raw_video/harmony4d}
OUT=$ROOT/extracted
mkdir -p "$OUT"

for z in $(find "$ROOT" -name "*.zip" | sort); do
  name=$(basename "$z" .zip)
  if [ -d "$OUT/$name" ]; then
    echo "== skip $name (Unzipped)"; continue
  fi
  echo "== unzip $name ($(du -h "$z" | cut -f1)) =="
  mkdir -p "$OUT/$name.tmp"
  unzip -q "$z" -d "$OUT/$name.tmp"
  mv "$OUT/$name.tmp" "$OUT/$name"
  if [ "${KEEP_ZIP:-0}" != "1" ]; then
    rm "$z" && echo "   zip Deleted (KEEP_ZIP=1 Can be reserved)"
  fi
  df -h "$ROOT" | tail -1
done

echo "===== Structural inventory (for ingest adaptation) ====="
INV=$ROOT/harmony4d_inventory.txt
{
  echo "# Harmony4D inventory $(date +%F)"
  du -sh "$OUT"/* 2>/dev/null
  echo "--- directory tree (depth <= 3 per package) ---"
  for d in "$OUT"/*/; do
    echo "## $d"
    find "$d" -maxdepth 3 | head -40
  done
  echo "--- file-type counts ---"
  find "$OUT" -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -15
} > "$INV"
echo "Inventory written to $INV"
