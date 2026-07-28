#!/usr/bin/env bash
# Pack local closed-source PNGs for a GitHub Release (maintainer helper).
# Default source pack: $MPIE_TEST_PACK or ~/mpie_testset_pack
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_PACK="${MPIE_TEST_PACK:-${1:-$HOME/mpie_testset_pack}}"
DEST="${2:-$ROOT/scratch/closed3_outputs.tar}"
MODELS=(gpt-image-2 gemini-3-pro-image seedream-5-pro)

mkdir -p "$(dirname "$DEST")"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for m in "${MODELS[@]}"; do
  if [[ ! -d "$SRC_PACK/outputs/$m" ]]; then
    echo "MISSING $SRC_PACK/outputs/$m" >&2
    exit 1
  fi
  echo "staging $m ..."
  mkdir -p "$tmp/$m"
  # Copy only PNGs (skip logs / pid / meta noise)
  find "$SRC_PACK/outputs/$m" -maxdepth 1 -type f -name '*.png' -print0 \
    | xargs -0 -I{} cp -n {} "$tmp/$m/"
done

tar -cf "$DEST" -C "$tmp" "${MODELS[@]}"
ls -lh "$DEST"
echo "Upload this archive as a GitHub Release asset, then set MPIE_CLOSED_OUTPUTS_URL."
