"""Real child processes prove failed and empty suites fail the caller."""

import pytest

from backend.testing import run_tests

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "body,exit_code,evidence",
    [
        (
            "def test_pass():\n    print('CHILD_STDOUT')\n    assert True\n",
            0,
            "CHILD_STDOUT",
        ),
        (
            "def test_fail():\n    import sys\n    print('CHILD_STDERR', file=sys.stderr)\n    assert False, 'intentional child failure'\n",
            1,
            "intentional child failure",
        ),
        ("# No tests here.\n", 5, "no tests ran"),
    ],
)
def test_runner_preserves_child_exit_code_and_diagnostic_transcript(
    tmp_path, body, exit_code, evidence
):
    (tmp_path / "test_child.py").write_text(body, encoding="utf-8")

    code = run_tests(
        ["--start-directory", str(tmp_path), "--log-dir", str(tmp_path / "logs")]
    )

    assert code == exit_code
    (log,) = (tmp_path / "logs").glob("*.log")
    transcript = log.read_text(encoding="utf-8")
    assert evidence in transcript
    assert "Started:" in transcript
    assert "Finished:" in transcript
    assert f"exit={exit_code}" in transcript


def test_runner_selects_only_the_requested_test_level(tmp_path):
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    unit\n    integration\n", encoding="utf-8"
    )
    (tmp_path / "test_levels.py").write_text(
        "import pytest\n@pytest.mark.unit\ndef test_unit():\n    assert True\n"
        "@pytest.mark.integration\ndef test_other_level():\n    assert False\n",
        encoding="utf-8",
    )

    code = run_tests(
        [
            "--start-directory",
            str(tmp_path),
            "--log-dir",
            str(tmp_path / "logs"),
            "--suite",
            "unit",
        ]
    )

    assert code == 0
    (log,) = (tmp_path / "logs").glob("*.log")
    assert "1 passed, 1 deselected" in log.read_text(encoding="utf-8")
