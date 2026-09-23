#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPA_DIR="${ROOT_DIR}/models/REPA"
DATA_DIR="${REPA_DATA_DIR:-${ROOT_DIR}/models/U-ViT/assets/datasets/coco256_features}"
OUTPUT_DIR="${REPA_OUTPUT_DIR:-${ROOT_DIR}/exps/repa}"

if [[ ! -d "${REPA_DIR}" ]]; then
  echo "REPA is missing. Run scripts/setup_repa.sh first." >&2
  exit 1
fi

if [[ ! -d "${DATA_DIR}/train" ]]; then
  echo "Preprocessed COCO features are missing at ${DATA_DIR}/train." >&2
  echo "Run scripts/download_coco2014.sh and scripts/setup_uvit_preprocessing.sh first." >&2
  exit 1
fi

cd "${REPA_DIR}"

accelerate launch train_t2i.py   --report-to="wandb"   --allow-tf32   --mixed-precision="fp16"   --seed=0   --path-type="linear"   --prediction="v"   --weighting="uniform"   --enc-type="dinov2-vit-b"   --proj-coeff=0.5   --encoder-depth=8   --output-dir="${OUTPUT_DIR}"   --exp-name="t2i_repa"   --data-dir="${DATA_DIR}"
