#!/usr/bin/env bash
set -euo pipefail
exec python3 -m backend.gpu.session_job "$@"
