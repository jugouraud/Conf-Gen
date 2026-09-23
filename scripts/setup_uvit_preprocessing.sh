#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UVIT_DIR="${ROOT_DIR}/models/U-ViT"
COCO_DIR="${COCO_DIR:-${ROOT_DIR}/data/coco}"

mkdir -p "${ROOT_DIR}/models"

if [[ ! -d "${UVIT_DIR}/.git" ]]; then
  git clone https://github.com/baofff/U-ViT.git "${UVIT_DIR}"
fi

mkdir -p "${UVIT_DIR}/assets/datasets"
rm -f "${UVIT_DIR}/assets/datasets/coco"
ln -s "${COCO_DIR}" "${UVIT_DIR}/assets/datasets/coco"

cat <<EOF
U-ViT preprocessing checkout is ready at:
  ${UVIT_DIR}

COCO is linked as:
  ${UVIT_DIR}/assets/datasets/coco -> ${COCO_DIR}

REPA follows U-ViT's MS-COCO preprocessing protocol. Before extracting
features, place U-ViT's Stable-Diffusion autoencoder directory at:

  ${UVIT_DIR}/assets/stable-diffusion/

It must contain:
  autoencoder_kl.pth

Then, from models/U-ViT, run:

  python scripts/extract_mscoco_feature.py --split train
  python scripts/extract_mscoco_feature.py --split val
  python scripts/extract_empty_feature.py

The resulting feature directory is:
  models/U-ViT/assets/datasets/coco256_features

REPA can train from that directory with:
  --data-dir="${UVIT_DIR}/assets/datasets/coco256_features"
EOF
