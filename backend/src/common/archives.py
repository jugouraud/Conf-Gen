"""Bounded, checksum-verified source and artifact transfer archives."""

import copy
import hashlib
import json
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from backend.src.common.paths import PROJECT_ROOT
from backend.src.common.storage import calculate_file_sha256


def collect_python_sources() -> dict[str, Path]:
    """Collect runtime code and experiment entry points, excluding generated runs."""
    paths = [PROJECT_ROOT / "backend/__init__.py"]
    for folder in ("backend/src", "backend/runtime"):
        paths.extend((PROJECT_ROOT / folder).rglob("*.py"))
    paths.extend((PROJECT_ROOT / "backend/experiments").glob("*/*.py"))
    return {path.relative_to(PROJECT_ROOT).as_posix(): path for path in paths}


@dataclass(frozen=True)
class ArchiveLimits:
    """Disk/expansion budgets for the retained experiment's transfer bundles."""

    max_members: int = 50_000
    max_manifest_bytes: int = 16 * 1024**2
    max_member_bytes: int = 8 * 1024**3
    max_total_bytes: int = 128 * 1024**3

    def __post_init__(self):
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.max_members,
                self.max_manifest_bytes,
                self.max_member_bytes,
                self.max_total_bytes,
            )
        ):
            raise ValueError("Archive limits must be positive integers")


def resolve_contained_artifact_path(root, relative):
    """Input: root and archive member. Output: contained path or ValueError."""
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in relative
        or ":" in relative
    ):
        raise ValueError(f"Invalid artifact path: {relative}")
    target = (Path(root) / relative).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Artifact escapes destination: {relative}")
    return target


def _benchmark_scientific_identity(report):
    """Exclude execution timings while retaining the model/field equality evidence."""
    if (
        not isinstance(report, dict)
        or report.get("passed") is not True
        or not report.get("bank_sha256")
        or not report.get("adapter_signature")
        or not report.get("acceptance")
        or not isinstance(report.get("batch_size"), int)
        or report["batch_size"] < 1
        or not report.get("records")
    ):
        raise ValueError("Invalid benchmark receipt for refresh")
    identity = copy.deepcopy(report)
    identity.pop("encoding_seconds", None)
    ids = []
    for record in identity["records"]:
        if (
            not record.get("id")
            or record.get("bitwise_equal") is not True
            or record.get("embedding_bitwise_equal") is not True
            or record.get("saved_checkpoint_bitwise_equal", True) is not True
            or record.get("max_absolute_error") != 0
            or record.get("relative_l2_error") != 0
        ):
            raise ValueError("Unverified benchmark receipt for refresh")
        ids.append(record["id"])
        for key in (
            "baseline_seconds",
            "optimized_field_seconds",
            "saved_checkpoint_bitwise_equal",
        ):
            record.pop(key, None)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate benchmark prompt identity")
    return identity


def extract_verified_archive(
    archive,
    root,
    manifest_name="artifact_manifest.json",
    *,
    replace_names=(),
    refresh_benchmarks=(),
    expected_hashes=None,
    limits: ArchiveLimits = ArchiveLimits(),
):
    """Input: manifested ZIP and destination. Output: verified files, refusing conflicts.

    Callers may explicitly name live status/log files that can be refreshed.
    Benchmark timings can be refreshed only with identical successful scientific
    evidence; the previous receipt is retained by hash. Frozen artifacts stay strict.
    """
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if (
            len(members) > limits.max_members
            or sum(info.file_size for info in members) > limits.max_total_bytes
        ):
            raise ValueError("Archive exceeds member count or expanded size limit")
        if any(
            info.file_size > limits.max_member_bytes
            or stat.S_ISLNK(info.external_attr >> 16)
            for info in members
        ):
            raise ValueError("Archive member exceeds size limit or is a symlink")
        if source.getinfo(manifest_name).file_size > limits.max_manifest_bytes:
            raise ValueError("Archive manifest exceeds size limit")
        manifest = json.loads(source.read(manifest_name))
        if not isinstance(manifest, dict) or any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            for name, digest in manifest.items()
        ):
            raise ValueError("Invalid artifact manifest")
        benchmark_history = {}
        names = source.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest) | {
            manifest_name
        }:
            raise ValueError("Unexpected archive contents")
        if expected_hashes is not None:
            if any(
                manifest.get(name) != digest for name, digest in expected_hashes.items()
            ):
                raise ValueError("Recovered archive differs from frozen inputs")
            manifest = dict(expected_hashes)
        # Validate every byte before committing the archive's files.
        for relative, digest in manifest.items():
            target = resolve_contained_artifact_path(root, relative)
            with source.open(relative) as reader:
                measured = hashlib.file_digest(reader, "sha256").hexdigest()
            if measured != digest:
                raise ValueError(f"Artifact checksum mismatch: {relative}")
            if target.exists() and calculate_file_sha256(target) != digest:
                if relative in refresh_benchmarks:
                    previous = json.loads(target.read_text(encoding="utf-8"))
                    incoming = json.loads(source.read(relative))
                    if _benchmark_scientific_identity(
                        previous
                    ) != _benchmark_scientific_identity(incoming):
                        raise ValueError(f"Incompatible benchmark receipt: {relative}")
                    old_hash = calculate_file_sha256(target)
                    original = PurePosixPath(relative)
                    historical = original.with_name(
                        f"{original.stem}.{old_hash}{original.suffix}"
                    )
                    history = resolve_contained_artifact_path(
                        root, f"gpu/benchmark_history/{historical}"
                    )
                    if history.exists() and calculate_file_sha256(history) != old_hash:
                        raise ValueError("Conflicting historical benchmark receipt")
                    benchmark_history[relative] = history
                elif relative not in replace_names:
                    raise ValueError(f"Conflicting existing artifact: {relative}")
        for relative in manifest:
            target = resolve_contained_artifact_path(root, relative)
            if target.exists() and calculate_file_sha256(target) == manifest[relative]:
                continue
            if relative in benchmark_history:
                history = benchmark_history[relative]
                history.parent.mkdir(parents=True, exist_ok=True)
                if not history.exists():
                    history.write_bytes(target.read_bytes())
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".transfer")
            with source.open(relative) as reader, temporary.open("wb") as writer:
                import shutil

                shutil.copyfileobj(reader, writer)
            temporary.replace(target)
    return manifest


def create_manifested_archive(
    path: Path, files: dict[str, Path], expected_hashes: dict[str, str] | None = None
) -> dict[str, str]:
    """Hash each input during ZIP creation and atomically commit verified bytes."""
    expected_hashes = {} if expected_hashes is None else expected_hashes
    if not set(expected_hashes).issubset(files):
        raise ValueError("Expected hashes include unstaged artifacts")
    for relative in files:
        resolve_contained_artifact_path(Path.cwd(), relative)
        if relative == "artifact_manifest.json":
            raise ValueError("Artifact manifest name is reserved")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as temporary_file:
        temporary = Path(temporary_file.name)
    try:
        with zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            for relative, local in files.items():
                digest = hashlib.sha256()
                with (
                    Path(local).open("rb") as reader,
                    archive.open(relative, "w", force_zip64=True) as writer,
                ):
                    while block := reader.read(8 * 1024**2):
                        digest.update(block)
                        writer.write(block)
                manifest[relative] = digest.hexdigest()
                if (
                    relative in expected_hashes
                    and manifest[relative] != expected_hashes[relative]
                ):
                    raise ValueError(f"Stale or corrupt staged artifact: {relative}")
            archive.writestr(
                "artifact_manifest.json", json.dumps(manifest, sort_keys=True)
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest
