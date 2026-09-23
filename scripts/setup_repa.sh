#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPA_DIR="${ROOT_DIR}/models/REPA"
REPA_REPO="https://github.com/sihyun-yu/REPA.git"
REPA_COMMIT="67f714503e3892f993844aab088ffc5791c92613"

mkdir -p "${ROOT_DIR}/models"

if [[ ! -d "${REPA_DIR}/.git" ]]; then
  git clone "${REPA_REPO}" "${REPA_DIR}"
fi

git -C "${REPA_DIR}" fetch --all --tags
git -C "${REPA_DIR}" checkout "${REPA_COMMIT}"

# Upstream consistency fix:
# dataset.MSCOCOFeatureDataset returns (raw_image, vae_moments, context), while
# train_t2i.py at the pinned revision attempts to unpack an unused fourth value.
python - "${REPA_DIR}/train_t2i.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "for raw_image, x, context, raw_captions in train_dataloader:"
new = "for raw_image, x, context in train_dataloader:"
if old in text:
    path.write_text(text.replace(old, new))
elif new not in text:
    raise SystemExit("Could not locate the expected REPA dataloader line; inspect upstream changes.")
PY

echo "REPA checked out and patched at:"
git -C "${REPA_DIR}" rev-parse HEAD

cat <<'EOF'

Create a dedicated environment for REPA to avoid dependency conflicts:

  python -m venv .venv-repa
  source .venv-repa/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r models/REPA/requirements.txt

The text-to-image entry point is:
  models/REPA/train_t2i.py
EOF
