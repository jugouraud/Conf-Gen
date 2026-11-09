"""Launch paired repair, collect every completed case, and release finished compute."""

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from backend.runtime.colab import ColabConnection
from backend.runtime.repair_colab import RepairColabConnection
from backend.src.common.archives import extract_verified_archive
from backend.src.common.storage import calculate_file_sha256, read_json, write_json

BOOTSTRAP = """import importlib.metadata,json,os,subprocess,sys,traceback
from pathlib import Path
root=Path(__file__).resolve().parent;output=root/'results'
status=output/'gpu_worker_status.json'
def update(**state):
 temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(state));temporary.replace(status)
try:
 update(status='running',phase='checking_dependencies')
 import tomllib
 from packaging.requirements import Requirement
 requirements=tomllib.loads((root/'pyproject.toml').read_text())['project']['dependencies']
 missing=[]
 for spec in requirements:
  requirement=Requirement(spec)
  if requirement.marker and not requirement.marker.evaluate():continue
  try:version=importlib.metadata.version(requirement.name)
  except importlib.metadata.PackageNotFoundError:version=None
  if version is None or version not in requirement.specifier:
   if requirement.name=='torch':raise RuntimeError('Refusing a CUDA stack replacement')
   missing.append(spec)
 if missing:subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
 update(status='running',phase='starting_repair')
 manifest=json.loads((output/'experiment.json').read_text())
 optimized=manifest.get('fingerprint',{}).get('settings',{}).get('execution',{}).get('optimized_gpu',False)
 module='backend.runtime.repair_cuda' if optimized else 'backend.runtime.repair_worker'
 subprocess.run([sys.executable,'-u','-m',module,'--output',str(output)],cwd=root,check=True)
except BaseException:
 if not status.exists() or json.loads(status.read_text()).get('status')!='failed':
  update(status='failed',error=traceback.format_exc())
 raise
"""


def connection_for(output):
    settings = (
        read_json(output / "experiment.json").get("fingerprint", {}).get("settings", {})
    )
    if settings.get("execution", {}).get("optimized_gpu"):
        return RepairColabConnection(output)
    return ColabConnection(output)


def launch(output):
    connection = connection_for(output)
    validation = read_json(output / "validation.json")
    if (
        calculate_file_sha256(
            Path(validation.get("package_path", output.parent / "gpu-inputs.zip"))
        )
        != validation["package_sha256"]
    ):
        raise ValueError("Prepared bundle changed")
    state = connection.preflight()
    if not state["session_exists"]:
        connection.create()
    connection.stage()
    response = connection.execute(f"""import json,os,subprocess,sys
from pathlib import Path
root=Path({connection.root!r});output=root/'results';pid_path=output/'worker.pid.json'
if pid_path.exists():
 pid=json.loads(pid_path.read_text())['pid'];process=Path('/proc')/str(pid)
 if process.exists() and str(root).encode() in (process/'cmdline').read_bytes():raise RuntimeError('Repair worker already alive')
worker=root/'paired_repair_bootstrap.py';worker.write_text({BOOTSTRAP!r})
env=os.environ.copy();env.update(PYTHONPATH=str(root),PYTHONUNBUFFERED='1',USE_TF='0',MPLBACKEND='Agg',HF_TOKEN_PATH={connection.settings["hf_token_path"]!r})
with (output/'gpu_worker.log').open('a') as log:
 process=subprocess.Popen([sys.executable,'-u',str(worker)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pid_path.write_text(json.dumps({{'pid':process.pid}}))
print('REPAIR_LAUNCH='+json.dumps({{'status':'launched','pid':process.pid}}))
""")
    receipt = next(
        json.loads(line.split("REPAIR_LAUNCH=", 1)[1])
        for line in response.splitlines()
        if "REPAIR_LAUNCH=" in line
    )
    receipt.update(
        session=connection.session, launched_at=datetime.now(timezone.utc).isoformat()
    )
    gpu = output / "gpu"
    write_json(gpu / "colab_launch.json", receipt)
    command = [
        sys.executable,
        "-u",
        "-m",
        "backend.runtime.repair_session",
        "--output",
        str(output),
    ]
    with (
        (gpu / "monitor.stdout.log").open("ab") as stdout,
        (gpu / "monitor.stderr.log").open("ab") as stderr,
    ):
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    write_json(
        gpu / "monitor.launch.json",
        {
            "pid": process.pid,
            "command": command,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {**receipt, "monitor_pid": process.pid}


def snapshot(connection):
    response = connection.execute(f"""import json,os
from pathlib import Path
output=Path({connection.root!r})/'results'
records={{}}
for name in ('gpu_worker_status.json','status.json','source/status.json','image_scoring/shieldgemma2/pilot/status.json'):
 path=output/name
 if path.exists():
  try:records[name]=json.loads(path.read_text())
  except ValueError:pass
exports=[json.loads(path.read_text()) for path in sorted((output/'exports').glob('*.json'))]
pid=json.loads((output/'worker.pid.json').read_text())['pid']
try:os.kill(pid,0);alive=True
except ProcessLookupError:alive=False
payload={{'records':records,'exports':exports,'worker_pid':pid,'worker_alive':alive}}
if records.get('gpu_worker_status.json',{{}}).get('status')=='failed' or not alive:
 log=output/'gpu_worker.log'
 if log.exists():
  with log.open('rb') as stream:
   stream.seek(max(0,log.stat().st_size-5000));payload['log_tail']=stream.read().decode(errors='replace')
print('REPAIR_SNAPSHOT='+json.dumps(payload))
""")
    return next(
        json.loads(line.split("REPAIR_SNAPSHOT=", 1)[1])
        for line in response.splitlines()
        if "REPAIR_SNAPSHOT=" in line
    )


def collect_export(connection, output, export):
    name = export["name"]
    if Path(name).name != name or not name.endswith(".zip"):
        raise ValueError("Invalid repair export name")
    receipt = output / "gpu/collected" / f"{name}.json"
    if receipt.exists():
        if read_json(receipt)["sha256"] != export["sha256"]:
            raise ValueError("Previously collected export changed")
        return
    with tempfile.TemporaryDirectory() as folder:
        archive = Path(folder) / name
        connection.invoke(
            "download",
            "-s",
            connection.session,
            f"{connection.root}/results/exports/{name}",
            str(archive),
            timeout=180,
        )
        if calculate_file_sha256(archive) != export["sha256"]:
            raise ValueError("Repair export checksum mismatch")
        destination = output
        if name.startswith("field_"):
            destination = (
                Path.home()
                / ".cache/thesis-paired-repair-checkpoints"
                / f"{output.parent.name}-{output.name}"
            )
            if not (output / "gpu/reference_cache.json").exists():
                write_json(
                    output / "gpu/reference_cache.json",
                    {
                        "directory": str(destination),
                        "purpose": "durable recovered reference fields",
                    },
                )
        extract_verified_archive(
            archive,
            destination,
            replace_names=("status.json", "gpu_worker_status.json"),
        )
    write_json(receipt, export)


def verify_completed_cases(output):
    manifest = read_json(output / "experiment.json")
    digest = calculate_file_sha256(output / "experiment.json")
    rows = []
    receipts = list((output / "pilot/records").glob("*.json"))
    if len(receipts) != manifest["expected_prompts"]:
        raise ValueError("Missing completed prompt receipts")
    expected = {row["id"] for row in manifest["selected_cohort"]}
    observed = set()
    for path in receipts:
        receipt = read_json(path)
        if receipt["experiment_sha256"] != digest:
            raise ValueError("Case belongs to a different experiment")
        for name, sha in receipt["artifacts"].items():
            if calculate_file_sha256(output / name) != sha:
                raise ValueError(f"Collected artifact changed: {name}")
        rows.extend(receipt["rows"])
        observed.update(row["id"] for row in receipt["rows"])
    if observed != expected or len(rows) != manifest["expected_generation_records"]:
        raise ValueError("Incomplete or unexpected generation records")
    options = manifest.get("fingerprint", {}).get("settings", {}).get("experiment", {})
    if options.get("cohort_selection") == "unsafe_test_only" and any(
        row["split"] != "unsafe_test" for row in rows
    ):
        raise ValueError("Safe test variants must not enter the unsafe-only experiment")
    if options.get("methods"):
        combinations = {
            (mode, method, seed)
            for mode in options["modes"]
            for method in options["methods"]
            for seed in options["generation_seeds"]
        }
        for prompt_id in expected:
            current = [
                (row["mode"], row["method"], row["seed"])
                for row in rows
                if row["id"] == prompt_id
            ]
            if len(current) != len(combinations) or set(current) != combinations:
                raise ValueError(
                    "Missing or duplicate prompt/method/mode/seed combinations"
                )
    scoring = read_json(output / "image_scoring/shieldgemma2/pilot/status.json")
    if scoring["status"] != "complete":
        raise ValueError("Independent image scoring incomplete")
    return len(rows)


def monitor(output):
    connection = connection_for(output)
    manifest = read_json(output / "experiment.json")
    if (
        manifest.get("fingerprint", {})
        .get("settings", {})
        .get("execution", {})
        .get("optimized_gpu")
    ):
        directory = (
            Path.home()
            / ".cache/thesis-paired-repair-checkpoints"
            / f"{output.parent.name}-{output.name}"
        )
        write_json(
            output / "gpu/reference_cache.json",
            {
                "directory": str(directory),
                "purpose": "durable recovered reference fields",
            },
        )
    state = {}
    collectors = ThreadPoolExecutor(max_workers=4)
    pending = {}
    collected = set()
    export_digests = {}
    while True:
        try:
            current = snapshot(connection)
            # Downloads continue between polls. A backlog or slow transfer must
            # never prevent publishing fresh worker status every polling cycle.
            for name, future in list(pending.items()):
                if future.done():
                    del pending[name]
                    future.result()
                    collected.add(name)
            for export in current["exports"]:
                name = export["name"]
                if name in export_digests and export_digests[name] != export["sha256"]:
                    raise ValueError("Previously observed export changed")
                export_digests[name] = export["sha256"]
                if name not in pending and name not in collected:
                    receipt = output / "gpu/collected" / f"{name}.json"
                    if (
                        receipt.exists()
                        and read_json(receipt)["sha256"] == export["sha256"]
                    ):
                        collected.add(name)
                    else:
                        pending[name] = collectors.submit(
                            collect_export, connection, output, export
                        )
            records = current["records"]
            worker = records.get("gpu_worker_status.json", {})
            detail = records.get("status.json", worker)
            if detail.get("phase") == "rebuild_probe_bank":
                detail = records.get("source/status.json", detail)
            completed = len(list((output / "pilot/records").glob("*.json")))
            state = {
                "status": "running",
                "stage": detail.get("phase", "starting"),
                "stage_report": detail,
                "completed_prompts": completed,
                "expected_prompts": manifest["expected_prompts"],
                "expected_pairs": manifest["expected_pairs"],
                "expected_generation_records": manifest["expected_generation_records"],
                "worker_pid": current["worker_pid"],
                "worker_alive": current["worker_alive"],
                "monitor_pid": os.getpid(),
                "session": connection.session,
                "error": None,
                "monitoring_error": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "poll_interval_seconds": 30,
                "collection": {
                    "available_exports": len(current["exports"]),
                    "collected_exports": len(collected),
                    "pending_exports": len(pending),
                },
            }
            if worker.get("status") == "failed" or not current["worker_alive"]:
                state.update(
                    status="error",
                    error=worker.get("error")
                    or current.get("log_tail")
                    or "Repair worker exited",
                )
            if worker.get("status") == "complete" and pending:
                state.update(status="collecting", stage="collecting_completed_results")
            elif worker.get("status") == "complete":
                if not (output / "gpu/collected/final.zip.json").exists():
                    raise ValueError("Final results not exported")
                state.update(
                    status="completed",
                    stage="completed",
                    generation_records=verify_completed_cases(output),
                )
                write_json(output / "progress.json", state)
                connection.invoke("stop", "-s", connection.session)
                sessions = connection.invoke("sessions")
                if f"[{connection.session}]" in sessions:
                    raise RuntimeError("Session still present after shutdown")
                write_json(
                    output / "gpu/shutdown.json",
                    {
                        "status": "terminated",
                        "after_verified_collection": True,
                        "at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                collectors.shutdown(wait=True)
                return state
            write_json(output / "progress.json", state)
            if state["status"] == "error":
                collectors.shutdown(wait=False, cancel_futures=True)
                return state
        except Exception as error:
            missing = f"Session '{connection.session}' not found" in str(error)
            state.update(
                status="error" if missing else "reconnecting",
                monitoring_error=str(error),
                worker_alive=None,
                monitor_pid=os.getpid(),
                session=connection.session,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            write_json(output / "progress.json", state)
            if missing:
                collectors.shutdown(wait=False, cancel_futures=True)
                return state
        time.sleep(30)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    monitor(parser.parse_args().output.resolve())
