import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from backend.storage.fairness import SafetyRecord, _create_engine, create_database
from frontend.plots import (
    EmbeddingVisualizationController,
    VisualizationData,
    load_visualization_data,
    reduce_embeddings,
)


class DatabaseEmbeddingVisualizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_directory = Path(self.temporary_directory.name)
        for image_name in ("one.jpg", "two.jpg"):
            (self.data_directory / image_name).touch()
        records_path = self.data_directory / "records.json"
        records_path.write_text(
            json.dumps(
                [
                    {"id": 1, "caption": "First", "image": "one.jpg"},
                    {"id": 2, "caption": "Second", "image": "two.jpg"},
                ]
            ),
            encoding="utf-8",
        )
        self.database_path = self.data_directory / "data.sqlite3"
        create_database(records_path=records_path, database_path=self.database_path)
        self._populate_records()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loads_flattened_embeddings_and_harmfulness_scores(self) -> None:
        data = load_visualization_data(self.database_path)

        np.testing.assert_array_equal(data.record_ids, np.array([1, 2]))
        np.testing.assert_array_equal(data.embeddings, np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32))
        np.testing.assert_array_equal(data.prompt_harmfulness, np.array([0.2, 0.4]))
        np.testing.assert_array_equal(data.image_harmfulness, np.array([0.6, 0.8]))

    def test_pca_projection_has_two_dimensions(self) -> None:
        projection = reduce_embeddings(np.array([[1, 2, 3], [3, 2, 1]], dtype=np.float32), "PCA")

        self.assertEqual(projection.shape, (2, 2))

    def test_includes_records_with_only_one_harmfulness_score(self) -> None:
        self._set_image_harmfulness(record_id=2, score=None)

        data = load_visualization_data(self.database_path)
        controller = EmbeddingVisualizationController(data)
        controller.set_score_field("Image harmfulness")
        figure = controller.build_figure()

        np.testing.assert_array_equal(data.record_ids, np.array([1, 2]))
        self.assertTrue(np.isnan(data.image_harmfulness[1]))
        self.assertEqual(figure["data"][1]["customdata"], [2])

    def test_3d_view_uses_harmfulness_as_the_height(self) -> None:
        data = VisualizationData(
            record_ids=np.array([1, 2]),
            embeddings=np.array([[1, 2, 3], [3, 2, 1]], dtype=np.float32),
            prompt_harmfulness=np.array([0.2, 0.4]),
            image_harmfulness=np.array([0.6, 0.8]),
        )
        controller = EmbeddingVisualizationController(data)
        controller.set_view_dimension("3D")
        figure = controller.build_figure()
        point_trace = figure["data"][1]

        self.assertEqual(point_trace["z"], data.prompt_harmfulness.tolist())
        self.assertEqual(point_trace["marker"]["colorscale"][-1], [1.0, "#500724"])
        self.assertEqual(figure["layout"]["scene"]["zaxis"]["title"], "Prompt harmfulness")
        self.assertEqual(figure["layout"]["scene"]["zaxis"]["range"], [0.0, 0.4])

    def _populate_records(self) -> None:
        embeddings = (np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([[5, 6], [7, 8]], dtype=np.float32))
        engine = _create_engine(self.database_path)
        with Session(engine) as session:
            for record_id, embedding, prompt_score, image_score in zip(
                (1, 2), embeddings, (0.2, 0.4), (0.6, 0.8), strict=True
            ):
                record = session.get(SafetyRecord, record_id)
                record.inner_embedding = embedding.tobytes()
                record.inner_embedding_shape = json.dumps(list(embedding.shape))
                record.prompt_harmfulness = prompt_score
                record.image_harmfulness = image_score
            session.commit()
        engine.dispose()

    def _set_image_harmfulness(self, record_id: int, score: float | None) -> None:
        engine = _create_engine(self.database_path)
        with Session(engine) as session:
            record = session.get(SafetyRecord, record_id)
            record.image_harmfulness = score
            session.commit()
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
