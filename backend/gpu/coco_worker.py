"""Remote worker for COCO caption text-context extraction on Google Colab."""

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

ARCHIVE_PATH = Path("/content/conf-gen-coco-job.zip")
WORK_DIRECTORY = Path("/content/conf-gen-coco-job")
OUTPUT_DATABASE_PATH = Path("/content/conf-gen-coco-output.sqlite3")


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The Colab runtime does not expose CUDA. Check the requested T4 allocation.")
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for item in WORK_DIRECTORY.iterdir():
        if item.name == "models":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        archive.extractall(WORK_DIRECTORY)
    ARCHIVE_PATH.unlink()
    manifest = json.loads((WORK_DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    os.environ["CONF_GEN_MODELS_DIR"] = str(WORK_DIRECTORY / "models")
    sys.path.insert(0, str(WORK_DIRECTORY))
    from backend.embeddings.coco import fill_coco_database

    fill_coco_database(
        annotations_path=WORK_DIRECTORY / manifest["annotations"],
        images_directory=WORK_DIRECTORY / manifest["images_directory"],
        database_path=WORK_DIRECTORY / "coco.sqlite3",
        model_id=manifest["model_id"],
        device="cuda",
    )
    shutil.copy2(WORK_DIRECTORY / "coco.sqlite3", OUTPUT_DATABASE_PATH)


if __name__ == "__main__":
    main()
