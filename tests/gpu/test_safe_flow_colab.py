"""Local, CPU-only tests of Colab result import security and integrity."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from backend.gpu.safe_flow_client import _check_path, merge_result


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SafeFlowColabTransferTests(unittest.TestCase):
    def test_invalid_result_paths(self):
        for bad in ("../evil", "/abs", "scores/../../evil", "features\\evil",
                    "models/weights", "scores/./file"):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                _check_path(bad)
        self.assertEqual(_check_path("scores/safe_test.csv"), "scores/safe_test.csv")

    def test_validated_merge_and_idempotence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "experiment"
            content = b"id,distance\n1,0.2\n"
            record = {"stage": "report", "corpus_sha256": "corpus-hash",
                      "config_sha256": "config-hash", "inputs": {},
                      "format_version": 1}
            result = dict(record, artifacts={"scores/safe_test.csv": sha(content)})
            archive = Path(td) / "received.zip"
            with zipfile.ZipFile(archive, "w") as writer:
                writer.writestr("result_manifest.json", json.dumps(result))
                writer.writestr("results/scores/safe_test.csv", content)
            self.assertEqual(merge_result(archive, root, record), 1)
            self.assertEqual((root / "scores/safe_test.csv").read_bytes(), content)
            self.assertEqual(merge_result(archive, root, record), 0)

    def test_checksum_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "experiment"
            record = {"stage": "all", "corpus_sha256": "hash",
                      "config_sha256": "hash", "inputs": {}, "format_version": 1}
            result = dict(record, artifacts={"scores/report.json": sha(b"expected")})
            archive = Path(td) / "received.zip"
            with zipfile.ZipFile(archive, "w") as writer:
                writer.writestr("result_manifest.json", json.dumps(result))
                writer.writestr("results/scores/report.json", b"tampered")
            with self.assertRaises(ValueError):
                merge_result(archive, root, record)
            self.assertFalse((root / "scores/report.json").exists())


if __name__ == "__main__":
    unittest.main()
