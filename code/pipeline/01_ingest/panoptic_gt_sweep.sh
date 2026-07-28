#!/bin/bash
# CMU Panoptic GT Count first: Only next Calibration+3DattitudeGT(~40M/sequence), Don’t download the video.
# Counting people/After contacting the density, decide which sequences to buy. HD video(~1.4G/aircraft seat), Avoid being blind.
# usage): nohup bash panoptic_gt_sweep.sh > ~/panoptic_sweep.log 2>&1 &
BASE=http://domedb.perception.cs.cmu.edu/webdata/dataset
ROOT=$HOME/mpie_data/datasets/panoptic
mkdir -p "$ROOT"

# candidate: haggling(3people socialize)33 / ultimatum·mafia(multiplayer game)9 / band(band)3 /
# dance·moonbaby(dance)9 / pizza·office 3。toddler/ian Not available to minors; pose/Instrument solos are not accepted.
SEQS="
170221_haggling_b1 170221_haggling_b2 170221_haggling_b3
170221_haggling_m1 170221_haggling_m2 170221_haggling_m3
170224_haggling_a1 170224_haggling_a2 170224_haggling_a3
170224_haggling_b1 170224_haggling_b2 170224_haggling_b3
170228_haggling_a1 170228_haggling_a2 170228_haggling_a3
170228_haggling_b1 170228_haggling_b2 170228_haggling_b3
170404_haggling_a1 170404_haggling_a2 170404_haggling_a3
170404_haggling_b1 170404_haggling_b2 170404_haggling_b3
170407_haggling_a1 170407_haggling_a2 170407_haggling_a3
170407_haggling_b1 170407_haggling_b2 170407_haggling_b3
160422_haggling1 160226_haggling1 160224_haggling1
160422_ultimatum1 160226_ultimatum1 160224_ultimatum1 160224_ultimatum2
160422_mafia2 160226_mafia1 160226_mafia2 160224_mafia1 160224_mafia2
160906_band1 160906_band2 160906_band3
170307_dance1 170307_dance2 170307_dance3 170307_dance4 170307_dance5 170307_dance6
160317_moonbaby1 160317_moonbaby2 160317_moonbaby3
160906_pizza1 170915_office1 170407_office2
"

for s in $SEQS; do
  d=$ROOT/$s; mkdir -p "$d"
  if [ ! -s "$d/calibration_$s.json" ]; then
    wget -q -O "$d/calibration_$s.json" "$BASE/$s/calibration_$s.json" || echo "MISS calib $s"
  fi
  if [ -d "$d/hdPose3d_stage1_coco19" ] || [ -d "$d/hdPose3d_stage1" ]; then
    echo "SKIP $s (AlreadyGT)"; continue
  fi
  if wget -q -O "$d/pose.tar" "$BASE/$s/hdPose3d_stage1_coco19.tar" && tar xf "$d/pose.tar" -C "$d"; then
    rm -f "$d/pose.tar"; echo "OK $s coco19"
  elif rm -f "$d/pose.tar" && wget -q -O "$d/pose.tar" "$BASE/$s/hdPose3d_stage1.tar" && tar xf "$d/pose.tar" -C "$d"; then
    rm -f "$d/pose.tar"; echo "OK $s legacy15"
  else
    rm -f "$d/pose.tar"; rmdir "$d" 2>/dev/null; echo "FAIL $s (noneGT)"
  fi
done
echo "sweep done. Next step: python panoptic_sweep_stats.py"
