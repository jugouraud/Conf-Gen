#!/usr/bin/env bash
set -euo pipefail

if python3 -c 'import numpy, sqlalchemy, tqdm, huggingface_hub' >/dev/null 2>&1; then
    exec python3 -m backend.validation.fill "$@"
fi

if command -v uv >/dev/null 2>&1; then
    exec uv run --no-project --with numpy --with sqlalchemy --with tqdm --with huggingface-hub \
        python -m backend.validation.fill "$@"
fi

echo "WSL Python needs numpy, sqlalchemy, tqdm, and huggingface-hub; install them or install uv." >&2
exit 1
