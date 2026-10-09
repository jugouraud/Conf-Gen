"""Run the safe-only representation comparison inside a Colab GPU session."""

import json
import os
import sys
import zipfile
from pathlib import Path

ARCHIVE = Path("/content/conf-gen-representation-job.zip")
WORK = Path("/content/conf-gen-representation-job")
RESULT = Path("/content/conf-gen-representation-result.zip")


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The Colab runtime does not expose CUDA")
    WORK.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        archive.extractall(WORK)
    manifest = json.loads((WORK / "manifest.json").read_text(encoding="utf-8"))
    os.environ["CONF_GEN_MODELS_DIR"] = str(WORK / "models")
    sys.path.insert(0, str(WORK))
    from backend.embeddings.compare_representations import compare

    compare(
        WORK / "corpus.parquet", WORK / "prompt_eos.sqlite3", WORK / "representations",
        candidates=tuple(manifest["candidates"]), device="cuda",
        batch_size=manifest["batch_size"], variant_count=manifest["variant_count"],
    )
    with zipfile.ZipFile(RESULT, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted((WORK / "representations").iterdir()):
            output.write(path, path.name)
        output.write(WORK / "prompt_eos.sqlite3", "prompt_eos.sqlite3")


if __name__ == "__main__":
    main()
