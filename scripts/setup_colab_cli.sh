#!/usr/bin/env bash

set -euo pipefail

readonly COLAB_CLI_VERSION="0.7.1"

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required to install uv." >&2
        exit 1
    fi

    local installer
    installer="$(mktemp)"
    if ! curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh -o "$installer"; then
        rm -f "$installer"
        exit 1
    fi
    sh "$installer"
    rm -f "$installer"
}

install_uv
export PATH="$HOME/.local/bin:$PATH"
uv tool install --upgrade "google-colab-cli==$COLAB_CLI_VERSION"
colab version
