"""Manage persistent local copies of Hugging Face model repositories."""

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from backend.paths import MODELS_ROOT

MODELS_DIRECTORY = Path(os.environ.get("CONF_GEN_MODELS_DIR", MODELS_ROOT))
MODEL_CONFIGURATION_FILE = "config.json"


def get_local_model_directory(
    model_id: str,
    *,
    token: str | None = None,
    models_directory: Path = MODELS_DIRECTORY,
) -> Path:
    """Return the local model directory, downloading the repository once if needed."""
    destination = models_directory / _model_directory_name(model_id)
    if (destination / MODEL_CONFIGURATION_FILE).is_file():
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=destination, token=token)
    return destination


def _model_directory_name(model_id: str) -> str:
    """Create a portable directory name that preserves the repository identity."""
    return model_id.replace("/", "--")
