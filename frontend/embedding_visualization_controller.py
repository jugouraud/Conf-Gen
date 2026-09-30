"""State and Plotly-figure generation for embedding visualization."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.create_dataset_database import DEFAULT_DATABASE_PATH, SafetyRecord, _create_engine


ProjectionMethod = Literal["PCA", "t-SNE"]
ScoreField = Literal["Prompt harmfulness", "Image harmfulness"]
ViewDimension = Literal["2D", "3D"]
DEFAULT_TSNE_PERPLEXITY = 30
RANDOM_SEED = 42
COLOUR_SCALE = [[0.0, "#fee2e2"], [0.5, "#b91c1c"], [1.0, "#500724"]]


@dataclass(frozen=True)
class VisualizationData:
    """Embeddings and harmfulness data aligned by database record ID."""

    record_ids: np.ndarray
    embeddings: np.ndarray
    prompt_harmfulness: np.ndarray
    image_harmfulness: np.ndarray


def load_visualization_data(database_path: str | Path = DEFAULT_DATABASE_PATH) -> VisualizationData:
    """Load records that have embeddings and at least one harmfulness score."""
    engine = _create_engine(database_path)
    try:
        with Session(engine) as session:
            records = session.scalars(
                select(SafetyRecord)
                .where(
                    SafetyRecord.inner_embedding.is_not(None),
                    SafetyRecord.inner_embedding_shape.is_not(None),
                    or_(SafetyRecord.prompt_harmfulness.is_not(None), SafetyRecord.image_harmfulness.is_not(None)),
                )
                .order_by(SafetyRecord.id)
            ).all()
    finally:
        engine.dispose()

    if len(records) < 2:
        raise ValueError("At least two records with embeddings and one harmfulness score are required.")

    embeddings = np.vstack([_decode_embedding(record) for record in records])
    return VisualizationData(
        record_ids=np.array([record.id for record in records]),
        embeddings=embeddings,
        prompt_harmfulness=_score_array(records, "prompt_harmfulness"),
        image_harmfulness=_score_array(records, "image_harmfulness"),
    )


def _score_array(records: list[SafetyRecord], field_name: str) -> np.ndarray:
    """Return a score array, using NaN to represent a score absent from the database."""
    scores = [getattr(record, field_name) for record in records]
    return np.array([score if score is not None else np.nan for score in scores])


def _decode_embedding(record: SafetyRecord) -> np.ndarray:
    """Restore one stored embedding into a flattened float32 feature vector."""
    if record.inner_embedding is None or record.inner_embedding_shape is None:
        raise ValueError(f"Record {record.id} does not have a stored inner embedding.")

    shape = _parse_embedding_shape(record.inner_embedding_shape, record.id)
    element_count = int(np.prod(shape))
    dtype = _infer_embedding_dtype(len(record.inner_embedding), element_count, record.id)
    return np.frombuffer(record.inner_embedding, dtype=dtype).reshape(shape).astype(np.float32, copy=False).ravel()


def _parse_embedding_shape(shape_value: str, record_id: int) -> tuple[int, ...]:
    """Parse and validate the JSON shape stored for an embedding."""
    try:
        dimensions = json.loads(shape_value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Record {record_id} has an invalid inner embedding shape.") from error
    if not isinstance(dimensions, list) or not dimensions or any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0 for dimension in dimensions
    ):
        raise ValueError(f"Record {record_id} has an invalid inner embedding shape.")
    return tuple(dimensions)


def _infer_embedding_dtype(byte_count: int, element_count: int, record_id: int) -> np.dtype:
    """Infer the historical database embedding dtype from its byte length."""
    if byte_count == element_count * np.dtype(np.float32).itemsize:
        return np.dtype(np.float32)
    if byte_count == element_count * np.dtype(np.float64).itemsize:
        return np.dtype(np.float64)
    raise ValueError(f"Record {record_id} has embedding bytes incompatible with its stored shape.")


def reduce_embeddings(embeddings: np.ndarray, method: ProjectionMethod) -> np.ndarray:
    """Project embeddings to two dimensions using the selected method."""
    if method == "PCA":
        return PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(embeddings)
    if method == "t-SNE":
        return TSNE(
            n_components=2,
            init="random",
            learning_rate="auto",
            perplexity=min(DEFAULT_TSNE_PERPLEXITY, len(embeddings) - 1),
            random_state=RANDOM_SEED,
        ).fit_transform(embeddings)
    raise ValueError(f"Unsupported projection method: {method}")


class EmbeddingVisualizationController:
    """Own visualization state and create serializable Plotly figures."""

    def __init__(self, data: VisualizationData) -> None:
        self.data = data
        self.projection_method: ProjectionMethod = "PCA"
        self.score_field: ScoreField = "Prompt harmfulness"
        self.view_dimension: ViewDimension = "2D"
        self._projections: dict[ProjectionMethod, np.ndarray] = {}

    def set_projection_method(self, method: str) -> None:
        """Select the two-dimensional projection method."""
        if method not in ("PCA", "t-SNE"):
            raise ValueError(f"Unsupported projection method: {method}")
        self.projection_method = method

    def set_view_dimension(self, dimension: str) -> None:
        """Select a flat or harmfulness-height visualization."""
        if dimension not in ("2D", "3D"):
            raise ValueError(f"Unsupported view dimension: {dimension}")
        self.view_dimension = dimension

    def set_score_field(self, score_field: str) -> None:
        """Select the harmfulness score used for colour and height."""
        if score_field not in ("Prompt harmfulness", "Image harmfulness"):
            raise ValueError(f"Unsupported harmfulness score: {score_field}")
        self.score_field = score_field

    def build_figure(self) -> dict[str, Any]:
        """Build a two-dimensional or harmfulness-height Plotly figure."""
        coordinates = self._get_projection()
        harmfulness = self._selected_harmfulness()
        if self.view_dimension == "3D":
            return self._build_3d_figure(coordinates, harmfulness)
        return self._build_2d_figure(coordinates, harmfulness)

    def _get_projection(self) -> np.ndarray:
        """Return the cached two-dimensional projection for the selected method."""
        if self.projection_method not in self._projections:
            self._projections[self.projection_method] = reduce_embeddings(self.data.embeddings, self.projection_method)
        return self._projections[self.projection_method]

    def _selected_harmfulness(self) -> np.ndarray:
        """Return the score array selected by the harmfulness toggle."""
        if self.score_field == "Prompt harmfulness":
            return self.data.prompt_harmfulness
        return self.data.image_harmfulness

    def _build_2d_figure(self, coordinates: np.ndarray, harmfulness: np.ndarray) -> dict[str, Any]:
        """Create a 2D scatter plot coloured by harmfulness."""
        return {
            "data": self._point_traces(coordinates, harmfulness, trace_type="scattergl"),
            "layout": self._base_layout(
                xaxis={"title": f"{self.projection_method} dimension 1"},
                yaxis={"title": f"{self.projection_method} dimension 2", "scaleanchor": "x", "scaleratio": 1},
            ),
        }

    def _build_3d_figure(self, coordinates: np.ndarray, harmfulness: np.ndarray) -> dict[str, Any]:
        """Create a 3D plot with 2D embedding coordinates and harmfulness as height."""
        maximum_harmfulness = _maximum_axis_height(harmfulness)
        point_traces = self._point_traces(coordinates, harmfulness, trace_type="scatter3d")
        rated_mask = np.isfinite(harmfulness)
        stem_trace = self._stem_trace(coordinates[rated_mask], harmfulness[rated_mask])
        return {
            "data": [stem_trace, *point_traces] if rated_mask.any() else point_traces,
            "layout": self._base_layout(
                scene={
                    "xaxis": {"title": f"{self.projection_method} dimension 1"},
                    "yaxis": {"title": f"{self.projection_method} dimension 2"},
                    "zaxis": {"title": self.score_field, "range": [0.0, maximum_harmfulness]},
                }
            ),
        }

    def _point_traces(self, coordinates: np.ndarray, harmfulness: np.ndarray, *, trace_type: str) -> list[dict[str, Any]]:
        """Create coloured scored points and gray points with a missing selected score."""
        rated_mask = np.isfinite(harmfulness)
        traces: list[dict[str, Any]] = []
        if rated_mask.any():
            traces.append(
                self._rated_marker_trace(
                    coordinates[rated_mask], harmfulness[rated_mask], self.data.record_ids[rated_mask], trace_type=trace_type
                )
            )
        if (~rated_mask).any():
            traces.append(self._unrated_marker_trace(coordinates[~rated_mask], self.data.record_ids[~rated_mask], trace_type=trace_type))
        return traces

    def _rated_marker_trace(
        self, coordinates: np.ndarray, harmfulness: np.ndarray, record_ids: np.ndarray, *, trace_type: str
    ) -> dict[str, Any]:
        """Create the coloured trace for points that have the selected harmfulness score."""
        trace: dict[str, Any] = {
            "type": trace_type,
            "mode": "markers",
            "x": coordinates[:, 0].tolist(),
            "y": coordinates[:, 1].tolist(),
            "customdata": record_ids.tolist(),
            "marker": {
                "size": 7,
                "color": harmfulness.tolist(),
                "colorscale": COLOUR_SCALE,
                "cmin": 0.0,
                "cmax": 1.0,
                "colorbar": {"title": self.score_field},
            },
            "hovertemplate": "Record %{customdata}<br>Harmfulness %{marker.color:.3f}<extra></extra>",
        }
        if trace_type == "scatter3d":
            trace["z"] = harmfulness.tolist()
        return trace

    @staticmethod
    def _unrated_marker_trace(coordinates: np.ndarray, record_ids: np.ndarray, *, trace_type: str) -> dict[str, Any]:
        """Create a gray trace for points whose selected harmfulness score is absent."""
        trace: dict[str, Any] = {
            "type": trace_type,
            "mode": "markers",
            "x": coordinates[:, 0].tolist(),
            "y": coordinates[:, 1].tolist(),
            "customdata": record_ids.tolist(),
            "name": "Score not filled",
            "marker": {"size": 7, "color": "#9ca3af"},
            "hovertemplate": "Record %{customdata}<br>Harmfulness not filled<extra></extra>",
        }
        if trace_type == "scatter3d":
            trace["z"] = [0.0] * len(record_ids)
        return trace

    def _stem_trace(self, coordinates: np.ndarray, harmfulness: np.ndarray) -> dict[str, Any]:
        """Create vertical stems that make each 3D point a harmfulness peak."""
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        z_values: list[float | None] = []
        for x_coordinate, y_coordinate, score in zip(coordinates[:, 0], coordinates[:, 1], harmfulness, strict=True):
            x_values.extend((float(x_coordinate), float(x_coordinate), None))
            y_values.extend((float(y_coordinate), float(y_coordinate), None))
            z_values.extend((0.0, float(score), None))
        return {
            "type": "scatter3d",
            "mode": "lines",
            "x": x_values,
            "y": y_values,
            "z": z_values,
            "line": {"color": "#7f1d1d", "width": 2},
            "hoverinfo": "skip",
        }

    def _base_layout(self, **axes: dict[str, Any]) -> dict[str, Any]:
        """Create shared layout settings for a readable responsive Plotly figure."""
        return {
            "title": self._plot_title(),
            "template": "plotly_white",
            "margin": {"l": 20, "r": 20, "t": 55, "b": 20},
            "uirevision": f"{self.projection_method}-{self.view_dimension}",
            **axes,
        }

    def _plot_title(self) -> str:
        """Describe the current display without implying a 3D embedding reduction."""
        if self.view_dimension == "3D":
            return f"{self.projection_method}: 2D embedding projection with {self.score_field} as height"
        return f"{self.projection_method}: 2D embedding projection coloured by {self.score_field}"


def _maximum_axis_height(harmfulness: np.ndarray) -> float:
    """Return a non-degenerate Z-axis maximum based on the selected database scores."""
    rated_scores = harmfulness[np.isfinite(harmfulness)]
    if not len(rated_scores):
        return 1.0
    maximum_score = float(np.max(rated_scores))
    return maximum_score if maximum_score > 0.0 else 1.0
