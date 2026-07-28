#!/usr/bin/env bash
# Download optional closed-source PNG baselines into data/testset/outputs/.
#
# Source: Git LFS branch assets/closed3 on this repository
#   (gpt / gemini / seedream archives; gpt is split into two parts).
#
# Usage (from repo root):
#   bash scripts/fetch_closed_outputs.sh
#
# Override remote if needed:
#   MPIE_ASSETS_REMOTE=git@github.com:AnnLin0628/mpie-bench.git bash scripts/fetch_closed_outputs.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/testset/outputs"
BRANCH="${MPIE_ASSETS_BRANCH:-assets/closed3}"
REMOTE="${MPIE_ASSETS_REMOTE:-}"

if [[ -z "$REMOTE" ]]; then
  REMOTE="$(git -C "$ROOT" remote get-url origin 2>/dev/null || true)"
fi
if [[ -z "$REMOTE" ]]; then
  REMOTE="https://github.com/AnnLin0628/mpie-bench.git"
fi

if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
  echo "ERROR: git-lfs is required (https://git-lfs.com)." >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Cloning $REMOTE ($BRANCH) with Git LFS ..."
git -c filter.lfs.smudge= -c filter.lfs.required=false \
  clone --depth 1 --branch "$BRANCH" --single-branch "$REMOTE" "$tmp/repo"
git -C "$tmp/repo" lfs install --local
git -C "$tmp/repo" lfs pull

mkdir -p "$OUT"

# gpt is split into two parts
if [[ -f "$tmp/repo/gpt-image-2.tar.part00" ]]; then
  echo "Assembling gpt-image-2.tar ..."
  cat "$tmp/repo"/gpt-image-2.tar.part* > "$tmp/gpt-image-2.tar"
  mkdir -p "$OUT/gpt-image-2"
  tar -xf "$tmp/gpt-image-2.tar" -C "$OUT/gpt-image-2"
fi

for m in gemini-3-pro-image seedream-5-pro; do
  if [[ -f "$tmp/repo/${m}.tar" ]]; then
    echo "Extracting $m ..."
    mkdir -p "$OUT/$m"
    tar -xf "$tmp/repo/${m}.tar" -C "$OUT/$m"
  fi
done

echo "Done. Outputs under $OUT"
for m in gpt-image-2 gemini-3-pro-image seedream-5-pro; do
  if [[ -d "$OUT/$m" ]]; then
    n=$(find "$OUT/$m" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')
    echo "  $m  png=$n"
  else
    echo "  $m  MISSING"
  fi
done
