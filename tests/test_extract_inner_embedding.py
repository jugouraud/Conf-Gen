import unittest

from backend.extract_inner_embedding import _get_projected_image_embedding


class ProjectedImageEmbeddingTests(unittest.TestCase):
    def test_returns_tensor_directly_for_transformers_v4(self) -> None:
        tensor = object()

        self.assertIs(_get_projected_image_embedding(tensor), tensor)

    def test_reads_projected_pooler_output_for_transformers_v5(self) -> None:
        projected_tensor = object()
        model_output = type("ModelOutput", (), {"pooler_output": projected_tensor})()

        self.assertIs(_get_projected_image_embedding(model_output), projected_tensor)


if __name__ == "__main__":
    unittest.main()
