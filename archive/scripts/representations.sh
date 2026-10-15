#!/usr/bin/env bash
set -euo pipefail

if python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
    exec python3 -m backend.gpu.prompt_representation_client "$@"
fi

if command -v uv >/dev/null 2>&1; then
    exec uv run --no-project --with huggingface-hub python -m backend.gpu.prompt_representation_client "$@"
fi

echo "WSL Python needs huggingface-hub; install it or install uv." >&2
exit 1
