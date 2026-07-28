#!/bin/bash
# EgoHumans Certified download (rclone + user Google read only OAuth, Getting around anonymous quotas)
# prefix: ~/.config/rclone/rclone.conf Already prepared [gdrv] remote (token by user rclone authorize supply)
# usage: bash egohumans_fetch_rclone.sh   log: ~/egohumans_rclone.log
set -u
RC=~/.local/bin/rclone
DEST=$MPIE_ROOT/data/raw/egohumans
mkdir -p "$DEST"

declare -A FOLDERS=(
  [01_tagging]=1UuU7yfF1o6qjIgSSBtbmbzxOfB8RxCFd
  [04_basketball]=1WHmbzSBt4jVMp8POcX5gsWc17KC4m_Gg
)
for sub in 01_tagging 04_basketball; do
  echo "== $sub =="
  $RC copy "gdrv:" "$DEST/$sub" \
    --drive-root-folder-id "${FOLDERS[$sub]}" \
    --drive-acknowledge-abuse \
    --transfers 4 --checkers 4 --retries 10 --low-level-retries 20 \
    --stats 60s --stats-one-line -v
done
echo "== Size check =="
fail=0
while read -r want f; do
  have=$(stat -c%s "$DEST/$f" 2>/dev/null || echo 0)
  [ "$have" = "$want" ] || { echo "MISMATCH $f have=$have want=$want"; fail=1; }
done <<'EOF'
7371898232 01_tagging/001_tagging.tar.gz
10044211943 01_tagging/002_tagging.tar.gz
3831343203 01_tagging/003_tagging.tar.gz
1664674708 01_tagging/004_tagging.tar.gz
5337147937 01_tagging/005_tagging.tar.gz
4551084260 01_tagging/006_tagging.tar.gz
8503149124 01_tagging/007_tagging.tar.gz
2547920219 01_tagging/008_tagging.tar.gz
6721639539 01_tagging/009_tagging.tar.gz
5001175137 01_tagging/010_tagging.tar.gz
8812486425 01_tagging/011_tagging.tar.gz
2430348273 01_tagging/012_tagging.tar.gz
2213209801 01_tagging/013_tagging.tar.gz
2560889937 01_tagging/014_tagging.tar.gz
11539704694 04_basketball/001_basketball.tar.gz
5582293515 04_basketball/002_basketball.tar.gz
5383382478 04_basketball/003_basketball.tar.gz
1253952613 04_basketball/004_basketball.tar.gz
4585527909 04_basketball/005_basketball.tar.gz
1822377179 04_basketball/006_basketball.tar.gz
5461548164 04_basketball/007_basketball.tar.gz
3689396177 04_basketball/008_basketball.tar.gz
1837210808 04_basketball/009_basketball.tar.gz
2690474400 04_basketball/010_basketball.tar.gz
5389310465 04_basketball/011_basketball.tar.gz
4570150117 04_basketball/012_basketball.tar.gz
1687072879 04_basketball/013_basketball.tar.gz
2149011263 04_basketball/014_basketball.tar.gz
EOF
[ "$fail" = 0 ] && echo "egohumans rclone fetch done $(date)" || echo "There is an incomplete file, Rerun this script to continue uploading"
