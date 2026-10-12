"""CPU-only checks for reusable Colab session jobs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.gpu.session_job import _remote_path, run_job


class SessionJobTests(unittest.TestCase):
    def test_reuses_session_and_transfers_files_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.py"
            source.write_text("print('ok')", encoding="utf-8")
            output = root / "results" / "output.csv"
            calls = []

            def run(command):
                calls.append(command)
                if command[3] == "download" and ".gpu-job-" in command[6]:
                    Path(command[7]).write_text(json.dumps({"returncode": 0}), encoding="utf-8")

            with (patch("backend.gpu.session_job._find_colab_command", return_value="colab"),
                  patch("backend.gpu.session_job._run_command", side_effect=run),
                  patch("backend.gpu.session_job._start_session") as start,
                  patch("backend.gpu.session_job._stop_session") as stop):
                run_job(session="shared", uploads=[(source, "/content/input.py")],
                        commands=["python /content/input.py"],
                        downloads=[("/content/output.csv", output)])
            self.assertEqual([call[3] for call in calls],
                             ["upload", "exec", "download", "download"])
            self.assertTrue(output.parent.is_dir())
            start.assert_not_called()
            stop.assert_not_called()

    def test_remote_failure_is_reported_and_requested_stop_runs(self):
        def run(command):
            if command[3] == "download":
                Path(command[7]).write_text(
                    json.dumps({"returncode": 7, "command": "false"}), encoding="utf-8")

        with (patch("backend.gpu.session_job._find_colab_command", return_value="colab"),
              patch("backend.gpu.session_job._run_command", side_effect=run),
              patch("backend.gpu.session_job._start_session") as start,
              patch("backend.gpu.session_job._stop_session") as stop):
            with self.assertRaisesRegex(RuntimeError, "Remote command failed"):
                run_job(session="shared", uploads=[], commands=["false"], downloads=[],
                        create=True, stop=True)
        start.assert_called_once()
        stop.assert_called_once()

    def test_invalid_paths_fail_before_session_creation(self):
        self.assertEqual(_remote_path("/tmp/output"), "/tmp/output")
        for path in ("tmp/output", "/content/../output", "/content/./output"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                _remote_path(path)
        with patch("backend.gpu.session_job._find_colab_command") as find:
            with self.assertRaises(FileNotFoundError):
                run_job(session="shared", uploads=[(Path("missing.py"), "/content/x.py")],
                        commands=[], downloads=[], create=True)
            find.assert_not_called()


if __name__ == "__main__":
    unittest.main()
