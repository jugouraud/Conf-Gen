"""Run arbitrary file-based jobs in a named Colab GPU session."""

import argparse
import json
import tempfile
import uuid
from pathlib import Path

from backend.gpu.colab import (
    _colab_prefix, _find_colab_command, _run_command, _start_session, _stop_session,
)


# Hierarchy 3: atomic actions.

def _remote_path(value: str) -> str:
    if not value.startswith("/") or value.endswith("/") or "\\" in value:
        raise ValueError(f"Remote file path must be an absolute POSIX file path: {value}")
    if any(part in (".", "..") for part in value.split("/")):
        raise ValueError(f"Remote file path cannot contain . or ..: {value}")
    return value


def _instructions(commands: list[str], status_path: str) -> str:
    return (
        "import json\n"
        "import subprocess\n"
        "from pathlib import Path\n"
        f"commands = {commands!r}\n"
        "result = {'returncode': 0, 'command': None}\n"
        "try:\n"
        "    for command in commands:\n"
        "        print(f'$ {command}', flush=True)\n"
        "        completed = subprocess.run(command, shell=True, cwd='/content')\n"
        "        if completed.returncode:\n"
        "            result = {'returncode': completed.returncode, 'command': command}\n"
        "            break\n"
        "except BaseException as error:\n"
        "    result = {'returncode': -1, 'command': command, 'error': repr(error)}\n"
        "finally:\n"
        f"    Path({status_path!r}).write_text(json.dumps(result))\n"
    )


# Hierarchy 2: subworkflows.

def run_job(*, session: str, uploads: list[tuple[Path, str]],
            commands: list[str], downloads: list[tuple[str, Path]],
            auth: str = "oauth2", gpu: str = "T4", high_mem: bool = False,
            create: bool = False, stop: bool = False, timeout: int = 86400) -> None:
    if not session.strip():
        raise ValueError("A named --session is required")
    if timeout < 1:
        raise ValueError("--timeout must be positive")
    if not (create or uploads or commands or downloads or stop):
        raise ValueError("Specify --create, --upload, --run, --download, or --stop")
    for local, remote in uploads:
        if not local.is_file():
            raise FileNotFoundError(local)
        _remote_path(remote)
    for remote, _ in downloads:
        _remote_path(remote)

    colab = _find_colab_command()
    prefix = _colab_prefix(colab, auth)
    if create:
        _start_session(colab, auth, session, gpu, high_mem)
    try:
        for local, remote in uploads:
            _run_command(prefix + ["upload", "-s", session, str(local.resolve()), remote])

        if commands:
            with tempfile.TemporaryDirectory(prefix="colab_job_") as directory:
                temporary = Path(directory)
                status_remote = f"/content/.gpu-job-{uuid.uuid4().hex}.json"
                instructions = temporary / "instructions.py"
                status_local = temporary / "status.json"
                instructions.write_text(_instructions(commands, status_remote), encoding="utf-8")
                _run_command(prefix + ["exec", "-s", session, "--timeout", str(timeout),
                                       "-f", str(instructions)])
                _run_command(prefix + ["download", "-s", session,
                                       status_remote, str(status_local)])
                status = json.loads(status_local.read_text(encoding="utf-8"))
                if status.get("returncode") != 0:
                    raise RuntimeError(f"Remote command failed: {status}")

        for remote, local in downloads:
            local.parent.mkdir(parents=True, exist_ok=True)
            _run_command(prefix + ["download", "-s", session, remote, str(local.resolve())])
    finally:
        if stop:
            _stop_session(colab, auth, session)


# Hierarchy 1: workflow orchestration.

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Named session to create or reuse")
    parser.add_argument("--create", action="store_true", help="Create the session before this job")
    parser.add_argument("--stop", action="store_true", help="Stop the session after this job")
    parser.add_argument("--gpu", choices=("T4", "L4", "G4", "H100", "A100"), default="T4")
    parser.add_argument("--high-mem", action="store_true")
    parser.add_argument("--auth", choices=("oauth2", "adc"), default="oauth2")
    parser.add_argument("--upload", nargs=2, action="append", metavar=("LOCAL", "REMOTE"),
                        default=[], help="Upload a file to an absolute remote path; repeatable")
    parser.add_argument("--run", action="append", default=[], metavar="COMMAND",
                        help="Run a shell command in /content; repeatable, in order")
    parser.add_argument("--download", nargs=2, action="append", metavar=("REMOTE", "LOCAL"),
                        default=[], help="Download a result file; repeatable")
    parser.add_argument("--timeout", type=int, default=86400,
                        help="Execution timeout in seconds (default: 86400)")
    args = parser.parse_args()
    try:
        run_job(session=args.session,
                uploads=[(Path(local), remote) for local, remote in args.upload],
                commands=args.run,
                downloads=[(remote, Path(local)) for remote, local in args.download],
                auth=args.auth, gpu=args.gpu, high_mem=args.high_mem,
                create=args.create, stop=args.stop, timeout=args.timeout)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
