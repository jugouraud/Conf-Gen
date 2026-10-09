"""Run step 3 on the project's Google Colab CLI GPU workflow."""

import argparse
import hashlib
import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

from backend.gpu.client import (
    _colab_prefix, _find_colab_command, _run_command, _start_session, _stop_session,
)
from backend.paths import DATA_ROOT, PROJECT_ROOT

DEFAULT_CORPUS = DATA_ROOT / "prompts" / "confgen_prompts.parquet"
DEFAULT_CACHE = DATA_ROOT / "prompts" / "prompt_eos.sqlite3"
DEFAULT_OUTPUT = DATA_ROOT / "prompts" / "representations"
DEFAULT_CANDIDATES = ("eos_final", "mean_final", "eos_layer_4", "eos_layer_8")

REMOTE_ARCHIVE = "/content/conf-gen-representation-job.zip"
REMOTE_RESULT = "/content/conf-gen-representation-result.zip"


def run_colab_comparison(
    corpus: Path = DEFAULT_CORPUS, cache: Path = DEFAULT_CACHE,
    output_dir: Path = DEFAULT_OUTPUT, *, gpu: str = "T4", auth: str = "oauth2",
    session: str | None = None, candidates: tuple[str, ...] = DEFAULT_CANDIDATES,
    batch_size: int = 64, variant_count: int = 256,
) -> Path:
    if not corpus.is_file():
        raise FileNotFoundError(corpus)
    import re
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("Candidates must be nonempty and unique")
    for name in candidates:
        if name not in {"eos_final", "mean_final"} and not re.fullmatch(r"eos_layer_[1-9][0-9]*", name):
            raise ValueError(f"Unknown CLIP representation: {name}")
    if batch_size < 1 or variant_count < 1:
        raise ValueError("batch_size and variant_count must be positive")
    colab = _find_colab_command()
    active_session = session or f"conf-gen-repr-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory(prefix="conf_gen_repr_") as directory:
        temporary = Path(directory)
        archive = temporary / "job.zip"
        result = temporary / "result.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as job:
            job.write(corpus, "corpus.parquet")
            job.writestr("manifest.json", json.dumps({
                "candidates": candidates, "batch_size": batch_size,
                "variant_count": variant_count,
            }))
            if cache.is_file():
                job.write(cache, "prompt_eos.sqlite3")
            code_paths = (
                "backend/__init__.py", "backend/paths.py",
                "backend/storage/__init__.py", "backend/storage/models.py",
                "backend/embeddings/__init__.py", "backend/embeddings/clip.py",
                "backend/embeddings/define_region.py", "backend/embeddings/prompt_region.py",
                "backend/embeddings/compare_representations.py",
            )
            for relative in code_paths:
                path = PROJECT_ROOT / relative
                job.write(path, path.relative_to(PROJECT_ROOT).as_posix())
        started = False
        try:
            _start_session(colab, auth, active_session, gpu, False)
            started = True
            _run_command(_colab_prefix(colab, auth) + [
                "install", "-s", active_session, "numpy>=2.0.0", "pandas>=2.0.0",
                "pyarrow>=15.0.0", "scikit-learn>=1.8.0", "transformers>=4.51.0",
                "huggingface-hub>=0.24.0",
            ])
            _run_command(_colab_prefix(colab, auth) + [
                "upload", "-s", active_session, str(archive), REMOTE_ARCHIVE,
            ])
            _run_command(_colab_prefix(colab, auth) + [
                "exec", "-s", active_session, "--timeout", "86400", "-f",
                str(PROJECT_ROOT / "backend" / "gpu" / "prompt_representation_worker.py"),
            ])
            _run_command(_colab_prefix(colab, auth) + [
                "download", "-s", active_session, REMOTE_RESULT, str(result),
            ])
        finally:
            if started:
                _stop_session(colab, auth, active_session)
        with zipfile.ZipFile(result) as completed:
            expected = {"selection.json", "prompt_eos.sqlite3"}
            if not expected.issubset(completed.namelist()):
                raise ValueError("Colab result is incomplete")
            selection = json.loads(completed.read("selection.json"))
            if selection["selected"] not in candidates or "selected_final_report" not in selection:
                raise ValueError("Colab result has no finalized selection")
            digest = hashlib.sha256()
            with corpus.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            if selection.get("corpus_sha256") != digest.hexdigest():
                raise ValueError("Colab result was fitted on a different corpus")
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in completed.namelist():
                if name == "prompt_eos.sqlite3":
                    destination = cache
                elif name == "selection.json" or name in {f"{candidate}.{suffix}"
                       for candidate in candidates for suffix in ("npz", "json")}:
                    destination = output_dir / name
                else:
                    raise ValueError(f"Unexpected Colab result file: {name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with completed.open(name) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
        for name in candidates:
            path = output_dir / f"{name}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report.update(region=str((output_dir / f"{name}.npz").resolve()),
                          source=str(corpus.resolve()))
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            selection["candidates"][name]["region"] = str((output_dir / f"{name}.npz").resolve())
        selection["selected_final_report"] = json.loads(
            (output_dir / f"{selection['selected']}.json").read_text(encoding="utf-8")
        )
        (output_dir / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )
    return output_dir / "selection.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", choices=("T4", "L4", "G4", "H100", "A100"), default="T4")
    parser.add_argument("--auth", choices=("oauth2", "adc"), default="oauth2")
    parser.add_argument("--session")
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--variant-count", type=int, default=256)
    args = parser.parse_args()
    path = run_colab_comparison(
        args.corpus, args.cache, args.output_dir, gpu=args.gpu, auth=args.auth,
        session=args.session, candidates=tuple(args.candidates),
        batch_size=args.batch_size, variant_count=args.variant_count,
    )
    print(f"Downloaded safe-only representation selection: {path}")


if __name__ == "__main__":
    main()
