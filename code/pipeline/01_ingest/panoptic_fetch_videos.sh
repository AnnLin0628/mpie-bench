#!/bin/bash
# Panoptic HD Video purchasing: GT inventory(panoptic_sweep_stats.py 2026-07-10)selected8sequence × 5aircraft position.
# Camera positions are dispersed and indexed to ensure diverse viewing angles, 404 Automatically fill positions in the alternative pool; old sequence(1602xx)Probably none HD, Report ⚠。
# usage): nohup bash panoptic_fetch_videos.sh > ~/panoptic_videos.log 2>&1 &
BASE=http://domedb.perception.cs.cmu.edu/webdata/dataset
ROOT=$HOME/mpie_data/datasets/panoptic
# 160226_ultimatum1 GTdamage(77%ghosting,After deduplication only1people)Deprecated, Change 160422_ultimatum1(Actor pool on different days)
SEQS="160422_ultimatum1 160224_ultimatum2 160226_mafia2 160422_mafia2 160226_haggling1 170407_haggling_a2 160906_pizza1 160906_band2"
WANT=5
CAMS="0 6 12 18 24 3 9 15 21 27 1 7 13 19 25"

for s in $SEQS; do
  d=$ROOT/$s/hdVideos; mkdir -p "$d"
  got=$(ls "$d" 2>/dev/null | grep -c '\.mp4$')
  for c in $CAMS; do
    [ "$got" -ge "$WANT" ] && break
    cc=$(printf "%02d" "$c")
    f=$d/hd_00_$cc.mp4
    [ -s "$f" ] && continue
    if wget -q -c -O "$f" "$BASE/$s/videos/hd_shared_crf20/hd_00_$cc.mp4" && [ -s "$f" ]; then
      got=$((got+1)); echo "OK  $s cam$cc $(du -h "$f" | cut -f1)"
    else
      rm -f "$f"; echo "404 $s cam$cc"
    fi
  done
  [ "$got" -lt "$WANT" ] && echo "⚠ $s only got $got indivualHD"Airport"
done
echo "video fetch done"
