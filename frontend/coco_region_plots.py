"""Read cached 3D COCO region projections and build interactive Plotly figures."""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Literal

import numpy as np

from backend.embeddings.define_region import DEFAULT_DATABASE_PATH, DEFAULT_REGION_PATH
from backend.embeddings.project_region import (
    DEFAULT_PROJECTION_PATH,
    projection_cache_metadata,
    read_projection_metadata,
)
from frontend.coco_region_validation import ValidationResult

ProjectionMethod = Literal["PCA", "t-SNE"]
RANDOM_SEED = 42


@dataclass(frozen=True)
class RegionVisualizationData:
    """Sampled cached coordinates and scores from the full region."""

    pca_coordinates: np.ndarray
    tsne_coordinates: np.ndarray
    caption_ids: np.ndarray
    captions: list[str]
    is_reference: np.ndarray
    scores: np.ndarray
    radius: float
    total_reference: int
    total_calibration: int
    miscoverage: float
    pca_explained_variance_ratio: np.ndarray


def load_region_visualization_data(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    region_path: str | Path = DEFAULT_REGION_PATH,
    projection_path: str | Path = DEFAULT_PROJECTION_PATH,
    *, max_anchors: int = 1200, max_calibration: int = 200,
) -> RegionVisualizationData:
    """Take a reproducible display sample from the precomputed SQLite cache."""
    if max_anchors < 1 or max_calibration < 1:
        raise ValueError("Sample sizes must be positive.")
    database, artifact, cache = Path(database_path), Path(region_path), Path(projection_path)
    expected = projection_cache_metadata(database, artifact)
    if read_projection_metadata(cache) != expected:
        raise ValueError(
            f"COCO projection cache is missing or stale: {cache}. "
            "Run python -m backend.embeddings.project_region first."
        )
    with closing(sqlite3.connect(f"{cache.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        summary_row = connection.execute("SELECT value FROM metadata WHERE key = 'region_summary'").fetchone()
        if summary_row is None:
            raise ValueError("COCO projection cache is incomplete; rebuild it.")
        summary = json.loads(summary_row[0])
        ratios = summary.get("pca_explained_variance_ratio")
        if not isinstance(ratios, list) or len(ratios) != 3:
            raise ValueError("COCO projection cache lacks PCA variance; run python -m backend.embeddings.project_region to upgrade it.")
        rows = connection.execute(
            "SELECT caption_id, caption, is_reference, distance, "
            "pca_1, pca_2, pca_3, tsne_1, tsne_2, tsne_3 "
            "FROM captions ORDER BY caption_id"
        ).fetchall()

    reference_indices = np.fromiter((i for i, row in enumerate(rows) if row[2]), dtype=np.int64)
    calibration_indices = np.fromiter((i for i, row in enumerate(rows) if not row[2]), dtype=np.int64)
    if len(reference_indices) != summary["n_reference"] or len(calibration_indices) != summary["n_calibration"]:
        raise ValueError("COCO projection cache has incomplete caption rows; rebuild it.")
    rng = np.random.default_rng(RANDOM_SEED)
    selected_reference = np.sort(rng.choice(reference_indices, min(max_anchors, len(reference_indices)), replace=False))
    selected_calibration = np.sort(rng.choice(calibration_indices, min(max_calibration, len(calibration_indices)), replace=False))
    selected = [rows[int(index)] for index in np.concatenate((selected_reference, selected_calibration))]
    return RegionVisualizationData(
        pca_coordinates=np.asarray([row[4:7] for row in selected], dtype=np.float32),
        tsne_coordinates=np.asarray([row[7:10] for row in selected], dtype=np.float32),
        caption_ids=np.asarray([row[0] for row in selected], dtype=np.int64),
        captions=[row[1] for row in selected],
        is_reference=np.asarray([row[2] for row in selected], dtype=bool),
        scores=np.asarray([row[3] for row in selected], dtype=np.float32),
        radius=float(summary["radius"]),
        total_reference=int(summary["n_reference"]),
        total_calibration=int(summary["n_calibration"]),
        miscoverage=float(summary["miscoverage"]),
        pca_explained_variance_ratio=np.asarray(ratios, dtype=np.float64),
    )


class CocoRegionVisualizationController:
    """Switch between stored 3D projections and render membership labels."""

    def __init__(self, data: RegionVisualizationData) -> None:
        self.data = data
        self.projection_method: ProjectionMethod = "PCA"
        self.validation: ValidationResult | None = None
        self._sample_anchor_distances: np.ndarray | None = None

    def set_projection_method(self, method: str) -> None:
        if method not in ("PCA", "t-SNE"):
            raise ValueError(f"Unsupported projection method: {method}")
        self.projection_method = method

    def set_validation(self, result: ValidationResult) -> None:
        """Use the selected prompt as the distance and decision reference."""
        anchor_index = {int(caption_id): index for index, caption_id in enumerate(result.reference_caption_ids)}
        self._sample_anchor_distances = np.asarray([
            result.anchor_distances[anchor_index[int(caption_id)]]
            for caption_id in self.data.caption_ids[self.data.is_reference]
        ], dtype=np.float32)
        self.validation = result

    def clear_validation(self) -> None:
        self.validation = None
        self._sample_anchor_distances = None

    def build_figure(self) -> dict[str, Any]:
        """Show reference anchors and held-out points classified in full space."""
        method = self.projection_method
        coordinates = self.data.pca_coordinates if method == "PCA" else self.data.tsne_coordinates
        ratios = self.data.pca_explained_variance_ratio
        axis_titles = (
            [f"PC {i + 1} ({ratio:.2%} variance)" for i, ratio in enumerate(ratios)]
            if method == "PCA" else [f"t-SNE {i + 1}" for i in range(3)]
        )
        reference = self.data.is_reference
        inside = ~reference & (self.data.scores <= self.data.radius)
        outside = ~reference & ~inside
        if self.validation is None:
            traces = [
                self._trace(coordinates, reference, "Reference anchors", "#95a7b5", 2.5, 0.24),
                self._trace(coordinates, inside, "Held-out · inside", "#0d9488", 5, 0.85),
                self._trace(coordinates, outside, "Held-out · outside", "#e2583e", 6, 0.95),
            ]
            title = f"COCO caption region · {method}"
        else:
            traces = [
                self._colored_reference_trace(coordinates[reference]),
                self._trace(coordinates, ~reference, "Held-out COCO captions", "#94a3b8", 2.5, 0.2),
                self._validation_trace(method),
            ]
            decision = "BLOCK" if self.validation.blocked else "ALLOW"
            comparison = ">" if self.validation.blocked else "≤"
            title = f"{decision} · distance {self.validation.score:.2f} {comparison} radius {self.validation.radius:.2f}"
        return {
            "data": [trace for trace in traces if trace is not None],
            "layout": {
                "title": {"text": title, "font": {"size": 20, "color": "#17324d"}},
                "paper_bgcolor": "#f8fafc", "plot_bgcolor": "#f8fafc",
                "font": {"family": "Inter, Arial, sans-serif", "color": "#334155"},
                "margin": {"l": 0, "r": 0, "t": 48, "b": 0},
                "legend": {"orientation": "h", "y": 1.04, "x": 0, "font": {"size": 12}},
                "scene": {
                    "xaxis": {"title": axis_titles[0], "backgroundcolor": "#f8fafc"},
                    "yaxis": {"title": axis_titles[1], "backgroundcolor": "#f8fafc"},
                    "zaxis": {"title": axis_titles[2], "backgroundcolor": "#f8fafc"},
                    "aspectmode": "data", "camera": {"eye": {"x": 1.5, "y": 1.4, "z": 1.1}},
                },
                "uirevision": method,
            },
        }

    def _colored_reference_trace(self, points: np.ndarray) -> dict[str, Any]:
        """Color displayed anchors by their full-space distance from the prompt."""
        assert self.validation is not None and self._sample_anchor_distances is not None
        distances = self._sample_anchor_distances
        ids = self.data.caption_ids[self.data.is_reference]
        captions = [escape(caption[:160]) for caption, selected in zip(
            self.data.captions, self.data.is_reference, strict=True
        ) if selected]
        return {
            "type": "scatter3d", "mode": "markers", "name": "Anchors · distance from selected prompt",
            "x": points[:, 0].tolist(), "y": points[:, 1].tolist(), "z": points[:, 2].tolist(),
            "customdata": [[int(caption_id), caption, float(distance)] for caption_id, caption, distance in zip(
                ids, captions, distances, strict=True
            )],
            "marker": {
                "size": 4, "opacity": 0.8, "color": distances.tolist(),
                "colorscale": [[0, "#0d9488"], [0.67, "#f6b351"], [1, "#d94f3b"]],
                "cmin": 0, "cmax": self.validation.radius * 1.5,
                "colorbar": {"title": "Distance to prompt", "tickvals": [0, self.validation.radius],
                             "ticktext": ["0", f"Radius {self.validation.radius:.1f}"]},
            },
            "hovertemplate": "Anchor %{customdata[0]}<br>%{customdata[1]}<br>Distance to prompt %{customdata[2]:.2f}<extra></extra>",
        }

    def _validation_trace(self, method: ProjectionMethod) -> dict[str, Any]:
        """Make the selected prompt visible above the cloud in either view."""
        assert self.validation is not None
        point = self.validation.pca_coordinates if method == "PCA" else self.validation.tsne_coordinates
        decision = "BLOCK" if self.validation.blocked else "ALLOW"
        return {
            "type": "scatter3d", "mode": "markers", "name": f"Selected prompt · {decision}",
            "x": [float(point[0])], "y": [float(point[1])], "z": [float(point[2])],
            "marker": {"size": 12, "symbol": "diamond", "color": "#c33c2d" if self.validation.blocked else "#087f72",
                       "line": {"color": "#ffffff", "width": 2}},
            "customdata": [[escape(self.validation.prompt[:160]), self.validation.score, self.validation.radius]],
            "hovertemplate": "%{customdata[0]}<br>Distance %{customdata[1]:.2f} / radius %{customdata[2]:.2f}<extra>" + decision + "</extra>",
        }

    def _trace(
        self, coordinates: np.ndarray, mask: np.ndarray, name: str,
        color: str, size: float, opacity: float,
    ) -> dict[str, Any] | None:
        if not mask.any():
            return None
        points = coordinates[mask]
        ids = self.data.caption_ids[mask]
        captions = [escape(caption[:160]) for caption, selected in zip(self.data.captions, mask, strict=True) if selected]
        scores = self.data.scores[mask]
        return {
            "type": "scatter3d", "mode": "markers", "name": f"{name} ({len(points):,})",
            "x": points[:, 0].tolist(), "y": points[:, 1].tolist(), "z": points[:, 2].tolist(),
            "customdata": [[int(caption_id), caption, float(score)] for caption_id, caption, score in zip(ids, captions, scores, strict=True)],
            "marker": {"size": size, "color": color, "opacity": opacity},
            "hovertemplate": "Caption %{customdata[0]}<br>%{customdata[1]}<br>Distance %{customdata[2]:.2f}<extra>" + name + "</extra>",
        }
