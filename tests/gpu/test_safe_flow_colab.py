"""Local, CPU-only tests of Colab result import security and integrity."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from backend.gpu.safe_flow_client import _check_path, create_job, merge_result, run_colab_flow


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SafeFlowColabTransferTests(unittest.TestCase):
    def test_job_includes_flow_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            corpus = root / "corpus.parquet"
            config.write_text("[paths]\n", encoding="utf-8")
            corpus.write_bytes(b"placeholder")
            archive = root / "job.zip"
            create_job(archive, config, corpus, root / "results", "probes", False)
            with zipfile.ZipFile(archive) as bundle:
                names = set(bundle.namelist())
                extracted = root / "remote"
                bundle.extractall(extracted)
            self.assertTrue({
                "backend/flow_experiment.py",
                "backend/paths.py",
                "backend/flow/__init__.py",
                "backend/flow/common.py",
                "backend/flow/corpus.py",
                "backend/flow/model.py",
                "backend/flow/probes.py",
                "backend/flow/features.py",
                "backend/flow/metrics.py",
                "backend/flow/calibration.py",
                "backend/flow/evaluation.py",
                "backend/flow/report.py",
                "backend/flow/workflow.py",
                "scripts/build_prompt_dataset.py",
            }.issubset(names))
            imported = subprocess.run(
                [sys.executable, "-m", "backend.flow.workflow", "--help"],
                cwd=extracted, capture_output=True, text=True, check=False,
            )
            self.assertEqual(imported.returncode, 0, imported.stderr)

    def test_named_session_can_be_kept_or_reused(self):
        config = Path(__file__).resolve().parents[2] / "configs" / "safe_flow.toml"
        with (patch("backend.gpu.safe_flow_client._find_colab_command", return_value="colab"),
              patch("backend.gpu.safe_flow_client.create_job", return_value={}),
              patch("backend.gpu.safe_flow_client.merge_result", return_value=0),
              patch("backend.gpu.safe_flow_client._run_command"),
              patch("backend.gpu.safe_flow_client._start_session") as start,
              patch("backend.gpu.safe_flow_client._stop_session") as stop):
            run_colab_flow(config, stage="probes", session="shared", keep_session=True)
            start.assert_called_once()
            stop.assert_not_called()
            start.reset_mock()
            run_colab_flow(config, stage="reference", session="shared", reuse_session=True)
            start.assert_not_called()
            stop.assert_not_called()

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
