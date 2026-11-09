"""Run the CPU test suite with a dated transcript of stdout and stderr."""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from backend.src.common.paths import PROJECT_ROOT


def configure_test_run(argv):
    """Input: CLI flags. Output: test selection and transcript destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-directory", type=Path, default=PROJECT_ROOT / "tests")
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "tests/logs")
    parser.add_argument(
        "--suite",
        choices=("all", "unit", "integration", "contract", "e2e"),
        default="all",
    )
    parser.add_argument("--coverage", action="store_true")
    return parser.parse_args(argv)


def execute_logged_test_suite(options):
    """Input: test options. Output: exit code; stream and retain both terminal channels."""
    options.log_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(ZoneInfo("Europe/Zurich"))
    path = options.log_dir / f"{started:%Y%m%d-%H%M%S-%f}.log"
    command = [
        sys.executable,
        "-u",
        "-m",
        "pytest",
        str(options.start_directory),
        "-o",
        f"python_files={options.pattern}",
        "-v",
        "--capture=tee-sys",
    ]
    if options.suite != "all":
        command.extend(["-m", options.suite])
    if options.coverage:
        command.extend(["--cov", "--cov-report=term-missing"])
    with path.open("w", encoding="utf-8") as log:

        def write_line(line):
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()

        write_line(f"Started: {started.isoformat()}\nCommand: {command!r}\n\n")
        with subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            for line in process.stdout:
                write_line(line)
            code = process.wait()
        write_line(
            f"\nFinished: {datetime.now(ZoneInfo('Europe/Zurich')).isoformat()}; exit={code}\n"
        )
    print(f"Test log: {path}")
    return code


def run_tests(argv=None):
    """Input: optional CLI flags. Output: suite exit code, with a dated transcript."""
    options = configure_test_run(argv)
    return execute_logged_test_suite(options)


if __name__ == "__main__":
    raise SystemExit(run_tests())
