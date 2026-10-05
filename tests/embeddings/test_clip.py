import unittest

from backend.embeddings.clip import _get_projected_image_embedding, build_parser


class ProjectedImageEmbeddingTests(unittest.TestCase):
    def test_returns_tensor_directly_for_transformers_v4(self) -> None:
        tensor = object()

        self.assertIs(_get_projected_image_embedding(tensor), tensor)

    def test_reads_projected_pooler_output_for_transformers_v5(self) -> None:
        projected_tensor = object()
        model_output = type("ModelOutput", (), {"pooler_output": projected_tensor})()

        self.assertIs(_get_projected_image_embedding(model_output), projected_tensor)


class CommandLineTests(unittest.TestCase):
    def test_gpu_flag_requests_default_cuda_device(self) -> None:
        args = build_parser().parse_args(
            ["--prompt", "A test", "--image", "image.png", "--output-dir", "output", "--gpu"]
        )

        self.assertTrue(args.gpu)
        self.assertIsNone(args.device)

    def test_gpu_flag_cannot_be_combined_with_device(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["--prompt", "A test", "--image", "image.png", "--output-dir", "output", "--gpu", "--device", "cpu"]
            )


if __name__ == "__main__":
    unittest.main()
