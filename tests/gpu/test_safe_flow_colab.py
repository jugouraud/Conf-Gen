"""Local, CPU-only tests of Colab result import security and integrity."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from backend.gpu import safe_flow_worker
from backend.gpu.safe_flow_result_transfer import assemble_result, split_result
from backend.gpu.safe_flow_client import (
    _check_path, create_job, download_result, merge_result, run_colab_flow,
    snapshot_files, split_job_archive,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SafeFlowColabTransferTests(unittest.TestCase):
    def test_result_download_resumes_verified_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "result.zip"
            first, second = b"already present", b"download this"
            Path(f"{archive}.part0000").write_bytes(first)
            Path(f"{archive}.parts.json").write_text(json.dumps({
                "archive_size": len(first + second),
                "archive_sha256": sha(first + second),
                "parts": [
                    {"size": len(first), "sha256": sha(first)},
                    {"size": len(second), "sha256": sha(second)},
                ],
            }))

            def download(command):
                if command[-2].endswith(".part0001"):
                    Path(command[-1]).write_bytes(second)

            with patch("backend.gpu.safe_flow_client._run_command",
                       side_effect=download) as run:
                download_result("colab", "oauth2", "test-session", archive,
                                split_remote=False)
            self.assertEqual(archive.read_bytes(), first + second)
            self.assertEqual(run.call_count, 2)  # Manifest and missing part only.

    def test_worker_forwards_token_only_to_workflow_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            original = os.environ.get("HF_TOKEN")
            with (patch.object(safe_flow_worker, "WORK", Path(directory)),
                  patch.object(safe_flow_worker.subprocess, "run",
                               return_value=subprocess.CompletedProcess([], 0)) as run):
                safe_flow_worker.run_workflow(Path("config.toml"),
                                              Path("corpus.parquet"),
                                              Path("results"), "safe-test", "test-token")
            self.assertEqual(run.call_args.kwargs["env"]["HF_TOKEN"], "test-token")
            self.assertEqual(os.environ.get("HF_TOKEN"), original)

    def test_held_out_stage_packages_only_its_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = (
                "splits/safe_reference.jsonl", "splits/safe_test.jsonl",
                "probes/probe_bank.pt", "features/safe_reference/a.pt",
                "features/safe_validation/b.pt", "features/safe_test/c.pt",
                "features/unsafe_test/d.pt", "scores/threshold.json",
                "scores/safe_validation.csv", "scores/safe_test.csv",
                "scores/unsafe_test.csv",
            )
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
            safe = snapshot_files(root, "safe-test")
            self.assertEqual(set(safe), {
                "splits/safe_reference.jsonl", "splits/safe_test.jsonl",
                "probes/probe_bank.pt", "features/safe_reference/a.pt",
                "features/safe_test/c.pt", "scores/threshold.json",
            })
            unsafe = snapshot_files(root, "unsafe-test")
            self.assertIn("features/unsafe_test/d.pt", unsafe)
            self.assertNotIn("features/safe_test/c.pt", unsafe)
            report = snapshot_files(root, "report")
            self.assertEqual(set(report), {
                "scores/threshold.json", "scores/safe_test.csv",
                "scores/unsafe_test.csv",
            })

    def test_result_parts_verify_before_reassembly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "result.zip"
            payload = bytes(range(256)) * 10
            source.write_bytes(payload)
            manifest = split_result(source, part_bytes=41)
            received = root / "received.zip"
            for index in range(len(json.loads(manifest.read_text())["parts"])):
                shutil.copyfile(Path(f"{source}.part{index:04d}"),
                                Path(f"{received}.part{index:04d}"))
            assemble_result(manifest, received)
            self.assertEqual(received.read_bytes(), payload)
            received.unlink()
            Path(f"{received}.part0000").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Invalid result part"):
                assemble_result(manifest, received)
            self.assertFalse(received.exists())

    def test_chunked_job_reassembly_checks_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            payload = bytes(range(256)) * 10
            source.write_bytes(payload)
            parts, manifest = split_job_archive(source, root, part_bytes=37)
            remote = root / "remote"
            remote.mkdir()
            target = remote / "job.zip"
            for index, part in enumerate(parts):
                shutil.copyfile(part, Path(f"{target}.part{index:04d}"))
            shutil.copyfile(manifest, Path(str(target) + ".parts.json"))
            with (patch.object(safe_flow_worker, "ARCHIVE", target),
                  patch.object(safe_flow_worker, "PARTS_MANIFEST",
                               Path(str(target) + ".parts.json"))):
                safe_flow_worker.assemble_job_archive()
                self.assertEqual(target.read_bytes(), payload)
                self.assertFalse(safe_flow_worker.PARTS_MANIFEST.exists())

                target.unlink()
                shutil.copyfile(manifest, safe_flow_worker.PARTS_MANIFEST)
                for index, part in enumerate(parts):
                    shutil.copyfile(part, Path(f"{target}.part{index:04d}"))
                Path(f"{target}.part0000").write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "integrity check"):
                    safe_flow_worker.assemble_job_archive()
                self.assertFalse(target.exists())

    def test_client_import_needs_no_pandas_in_wsl(self):
        script = (
            "import builtins\n"
            "original = builtins.__import__\n"
            "def reject_pandas(name, *args, **kwargs):\n"
            "    if name == 'pandas' or name.startswith('pandas.'):\n"
            "        raise ModuleNotFoundError('pandas intentionally unavailable')\n"
            "    return original(name, *args, **kwargs)\n"
            "builtins.__import__ = reject_pandas\n"
            "import backend.gpu.safe_flow_client\n"
        )
        imported = subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(imported.returncode, 0, imported.stderr)

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
              patch("backend.gpu.safe_flow_client.upload_job"),
              patch("backend.gpu.safe_flow_client.download_result"),
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

    def test_remote_failure_reports_workflow_error(self):
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "result.zip"
            with zipfile.ZipFile(archive, "w") as writer:
                writer.writestr("error.json", json.dumps({
                    "error": "ValueError: Empty prompt in corpus",
                }))
            with self.assertRaisesRegex(RuntimeError, "Empty prompt in corpus"):
                merge_result(archive, Path(td) / "results", {})


if __name__ == "__main__":
    unittest.main()
