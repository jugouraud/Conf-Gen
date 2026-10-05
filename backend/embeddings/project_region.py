"""Precompute 3D COCO region views in a small SQLite cache.

Run ``python -m backend.embeddings.project_region`` after fitting the region.
The cache is separate from coco.sqlite3 because the saved region's source
signature includes that database's file size and modification time.
"""

import argparse
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from backend.embeddings.define_region import (
    DEFAULT_DATABASE_PATH,
    DEFAULT_REGION_PATH,
    _load_eos_vectors,
    _load_region,
    _nearest_distances,
    _source_signature,
)
from backend.paths import COCO_ROOT

DEFAULT_PROJECTION_PATH = COCO_ROOT / "coco_region_projections.sqlite3"
PROJECTION_VERSION = 1
RANDOM_SEED = 42


def _region_signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def projection_cache_metadata(database: Path, region_path: Path) -> dict:
    """Describe exactly which inputs a projection cache belongs to."""
    return {
        "version": PROJECTION_VERSION,
        "source": _source_signature(database),
        "region": _region_signature(region_path),
    }


def read_projection_metadata(path: str | Path) -> dict | None:
    """Return metadata for a complete cache, or None when unavailable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key = 'inputs'").fetchone()
            return json.loads(row[0]) if row else None
    except (sqlite3.Error, ValueError, TypeError):
        return None


def _ensure_pca_variance_summary(
    cache: Path, database: Path, artifact: Path, region, inputs: dict,
) -> None:
    """Upgrade an existing cache without recomputing its 3D t-SNE fit."""
    with closing(sqlite3.connect(f"{cache.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        summary = json.loads(connection.execute(
            "SELECT value FROM metadata WHERE key = 'region_summary'"
        ).fetchone()[0])
        if "pca_explained_variance_ratio" in summary:
            return
        rows = connection.execute(
            "SELECT caption_id, pca_1, pca_2, pca_3 FROM captions ORDER BY caption_id"
        ).fetchall()

    print("Calculating PCA explained variance for the existing cache...", flush=True)
    caption_ids, _, vectors = _load_eos_vectors(database, region.metadata["model_id"])
    if projection_cache_metadata(database, artifact) != inputs:
        raise RuntimeError("The COCO database or region changed while PCA variance was calculated.")
    if len(rows) != len(caption_ids) or any(row[0] != caption_id for row, caption_id in zip(rows, caption_ids, strict=True)):
        raise ValueError("Cached PCA coordinates do not match the COCO captions; rebuild the cache with --force.")
    coordinates = np.asarray([row[1:] for row in rows], dtype=np.float64)
    total_variance = float(np.var(region.whiten(vectors), axis=0, dtype=np.float64, ddof=1).sum())
    if total_variance <= 0:
        raise ValueError("COCO embeddings have no variance to explain.")
    summary["pca_explained_variance_ratio"] = (
        np.var(coordinates, axis=0, ddof=1) / total_variance
    ).tolist()
    if projection_cache_metadata(database, artifact) != inputs:
        raise RuntimeError("The COCO database or region changed while PCA variance was calculated.")
    with closing(sqlite3.connect(cache)) as connection:
        connection.execute("UPDATE metadata SET value = ? WHERE key = 'region_summary'", (json.dumps(summary),))
        connection.commit()


def _ensure_pca_model(cache: Path, database: Path, artifact: Path, region, inputs: dict) -> None:
    """Save the original PCA basis so a new prompt can be placed exactly."""
    with closing(sqlite3.connect(f"{cache.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        if connection.execute("SELECT 1 FROM metadata WHERE key = 'pca_model'").fetchone():
            return
        rows = connection.execute(
            "SELECT caption_id, pca_1, pca_2, pca_3 FROM captions ORDER BY caption_id"
        ).fetchall()

    print("Recovering the PCA transform for live prompts...", flush=True)
    caption_ids, _, vectors = _load_eos_vectors(database, region.metadata["model_id"])
    if projection_cache_metadata(database, artifact) != inputs:
        raise RuntimeError("The COCO database or region changed while PCA was fitted.")
    if len(rows) != len(caption_ids) or any(row[0] != caption_id for row, caption_id in zip(rows, caption_ids, strict=True)):
        raise ValueError("Cached PCA coordinates do not match the COCO captions; rebuild with --force.")
    whitened = region.whiten(vectors)
    pca = PCA(
        n_components=min(30, len(vectors) - 1, vectors.shape[1]),
        svd_solver="randomized", random_state=RANDOM_SEED,
    )
    reduced = pca.fit_transform(whitened)
    expected = np.asarray([row[1:] for row in rows], dtype=np.float32)
    # Randomized PCA can rotate slightly on a refit. Align all 30 recovered
    # coordinates to the stored three axes rather than relying on their signs.
    design = np.column_stack((reduced, np.ones(len(reduced), dtype=reduced.dtype)))
    alignment = np.linalg.lstsq(design, expected, rcond=None)[0]
    max_error = float(np.max(np.abs(design @ alignment - expected)))
    if max_error > 0.01:
        raise ValueError(f"Cached PCA coordinates could not be recovered (max error {max_error:.3g}); rebuild with --force.")
    if projection_cache_metadata(database, artifact) != inputs:
        raise RuntimeError("The COCO database or region changed while PCA was fitted.")
    payload = json.dumps({
        "mean": pca.mean_.tolist(),
        "components": (alignment[:-1].T @ pca.components_).tolist(),
        "offset": alignment[-1].tolist(),
    })
    with closing(sqlite3.connect(cache)) as connection:
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("pca_model", payload))
        connection.commit()


def project_coco_region(
    *, database_path: str | Path = DEFAULT_DATABASE_PATH,
    region_path: str | Path = DEFAULT_REGION_PATH,
    output_path: str | Path = DEFAULT_PROJECTION_PATH,
    force: bool = False,
) -> Path:
    """Store every caption's 3D PCA/t-SNE coordinates and true region score."""
    database, artifact, destination = Path(database_path), Path(region_path), Path(output_path)
    if destination.resolve() in {database.resolve(), artifact.resolve()}:
        raise ValueError("The projection cache output must be separate from the COCO database and region artifact.")
    region = _load_region(artifact)
    inputs = projection_cache_metadata(database, artifact)
    if region.metadata.get("source") != inputs["source"]:
        raise ValueError("The saved region does not match this COCO database. Refit it first.")
    if not force and read_projection_metadata(destination) == inputs:
        _ensure_pca_variance_summary(destination, database, artifact, region, inputs)
        _ensure_pca_model(destination, database, artifact, region, inputs)
        return destination

    print("Loading COCO EOS vectors...", flush=True)
    caption_ids, _, vectors = _load_eos_vectors(database, region.metadata["model_id"])
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        caption_rows = connection.execute("SELECT caption_id, caption FROM captions ORDER BY caption_id").fetchall()
    if len(caption_rows) != len(caption_ids) or any(row[0] != caption_id for row, caption_id in zip(caption_rows, caption_ids, strict=True)):
        raise ValueError("COCO caption text and embeddings are not aligned.")
    if projection_cache_metadata(database, artifact) != inputs:
        raise RuntimeError("The COCO database or region changed while embeddings were read; rerun the command.")
    if len(vectors) < 4 or vectors.shape[1] < 3:
        raise ValueError("At least four caption vectors with three dimensions are needed.")
    reference_ids = np.asarray(region.metadata["reference_caption_ids"], dtype=np.int64)
    if len(reference_ids) != len(region.anchors):
        raise ValueError("The saved region has inconsistent reference IDs and anchors.")
    reference_mask = np.isin(caption_ids, reference_ids)
    if int(reference_mask.sum()) != len(reference_ids):
        raise ValueError("The saved region's reference captions are missing from the database.")

    print(f"Reducing {len(vectors):,} captions with PCA and 3D t-SNE...", flush=True)
    whitened = region.whiten(vectors)
    pca = PCA(
        n_components=min(30, len(vectors) - 1, vectors.shape[1]),
        svd_solver="randomized", random_state=RANDOM_SEED,
    )
    reduced = pca.fit_transform(whitened)
    pca_3d = reduced[:, :3]
    tsne_3d = TSNE(
        n_components=3, init="pca", learning_rate="auto",
        perplexity=min(30, len(vectors) - 1), random_state=RANDOM_SEED,
    ).fit_transform(reduced)
    print("Scoring held-out captions against the full region...", flush=True)
    scores = np.zeros(len(vectors), dtype=np.float32)
    scores[~reference_mask] = _nearest_distances(whitened[~reference_mask], region.anchors)

    if projection_cache_metadata(database, artifact) != inputs:
        raise RuntimeError("The COCO database or region changed during projection; rerun the command.")
    print("Writing projection cache...", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, prefix=".coco_projections_", suffix=".sqlite3", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with closing(sqlite3.connect(temporary_path)) as connection:
            connection.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE captions (
                    caption_id INTEGER PRIMARY KEY,
                    caption TEXT NOT NULL,
                    is_reference INTEGER NOT NULL,
                    distance REAL NOT NULL,
                    pca_1 REAL NOT NULL, pca_2 REAL NOT NULL, pca_3 REAL NOT NULL,
                    tsne_1 REAL NOT NULL, tsne_2 REAL NOT NULL, tsne_3 REAL NOT NULL
                );
                CREATE INDEX ix_captions_role ON captions(is_reference);
            """)
            connection.execute("INSERT INTO metadata VALUES (?, ?)", ("inputs", json.dumps(inputs)))
            connection.execute("INSERT INTO metadata VALUES (?, ?)", ("region_summary", json.dumps({
                "radius": region.radius,
                "n_reference": len(reference_ids),
                "n_calibration": int((~reference_mask).sum()),
                "miscoverage": region.metadata["miscoverage"],
                "pca_explained_variance_ratio": pca.explained_variance_ratio_[:3].tolist(),
            })))
            connection.execute("INSERT INTO metadata VALUES (?, ?)", ("pca_model", json.dumps({
                "mean": pca.mean_.tolist(), "components": pca.components_[:3].tolist(),
                "offset": [0.0, 0.0, 0.0],
            })))
            connection.executemany(
                "INSERT INTO captions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (int(caption_id), caption_row[1], int(is_reference), float(score),
                     *map(float, pca), *map(float, tsne))
                    for caption_id, caption_row, is_reference, score, pca, tsne in zip(
                        caption_ids, caption_rows, reference_mask, scores, pca_3d, tsne_3d, strict=True
                    )
                ),
            )
            connection.commit()
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache 3D PCA/t-SNE coordinates for the saved COCO region.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--region", type=Path, default=DEFAULT_REGION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROJECTION_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = project_coco_region(
        database_path=args.database, region_path=args.region,
        output_path=args.output, force=args.force,
    )
    print(f"COCO 3D projection cache: {output}")


if __name__ == "__main__":
    main()
