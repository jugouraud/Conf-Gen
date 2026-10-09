#!/usr/bin/env bash
set -euo pipefail
if python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
    exec python3 -m backend.gpu.safe_flow_client "$@"
fi
if command -v uv >/dev/null 2>&1; then
    exec uv run --no-project --with huggingface-hub python -m backend.gpu.safe_flow_client "$@"
fi
echo "Install huggingface-hub in WSL Python or install uv." >&2
exit 1
