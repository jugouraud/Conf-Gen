#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COCO_DIR="${COCO_DIR:-${ROOT_DIR}/data/coco}"
DOWNLOAD_DIR="${COCO_DIR}/downloads"

mkdir -p "${DOWNLOAD_DIR}" "${COCO_DIR}"

download() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "Already present: $out"
    return
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "Need curl or wget." >&2
    exit 1
  fi
}

download "https://images.cocodataset.org/zips/train2014.zip" "${DOWNLOAD_DIR}/train2014.zip"
download "https://images.cocodataset.org/zips/val2014.zip" "${DOWNLOAD_DIR}/val2014.zip"
download "https://images.cocodataset.org/annotations/annotations_trainval2014.zip" "${DOWNLOAD_DIR}/annotations_trainval2014.zip"

unzip -q -n "${DOWNLOAD_DIR}/train2014.zip" -d "${COCO_DIR}"
unzip -q -n "${DOWNLOAD_DIR}/val2014.zip" -d "${COCO_DIR}"
unzip -q -n "${DOWNLOAD_DIR}/annotations_trainval2014.zip" -d "${COCO_DIR}"

echo
echo "COCO 2014 ready under: ${COCO_DIR}"
echo "Expected counts for REPA: 82,783 train images and 40,504 validation images."
