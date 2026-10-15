#!/usr/bin/env bash
set -euo pipefail

if python3 -c 'import numpy, tqdm, huggingface_hub' >/dev/null 2>&1; then
    exec python3 -m backend.embeddings.coco "$@"
fi

if command -v uv >/dev/null 2>&1; then
    exec uv run --no-project --with numpy --with tqdm --with huggingface-hub \
        python -m backend.embeddings.coco "$@"
fi

echo "WSL Python needs numpy, tqdm, and huggingface-hub; install them or install uv." >&2
exit 1
