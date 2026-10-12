"""Colab CLI session helpers used by the safe-flow client."""

import shutil
import subprocess
from collections.abc import Sequence


# Hierarchy 3: atomic actions.

def _find_colab_command() -> str:
    colab_command = shutil.which("colab")
    if colab_command is None:
        raise RuntimeError(
            "The Colab CLI is not installed. Install it on Linux, macOS, or WSL with "
            "`uv tool install google-colab-cli`."
        )
    return colab_command


def _colab_prefix(colab_command: str, auth: str) -> list[str]:
    return [colab_command, "--auth", auth]


def _run_command(command: Sequence[str], *, timeout: int | None = None) -> None:
    subprocess.run(command, check=True, timeout=timeout)


# Hierarchy 2: subworkflows.

def _start_session(colab_command: str, auth: str, session_name: str, gpu: str, high_memory: bool) -> None:
    command = _colab_prefix(colab_command, auth) + ["new", "-s", session_name, "--gpu", gpu]
    if high_memory:
        command.append("--high-mem")
    _run_command(command)


def _stop_session(colab_command: str, auth: str, session_name: str) -> None:
    try:
        _run_command(_colab_prefix(colab_command, auth) + ["stop", "-s", session_name])
    except subprocess.CalledProcessError as error:
        print(f"Warning: Colab session cleanup failed: {error}")
