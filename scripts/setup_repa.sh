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

echo "REPA checked out at:"
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
