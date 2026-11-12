"""Bounded procedural jobs or subprocess optimization with atomic receipts."""

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from backend.src.expert.config import ExpertConfig, Guidance
from backend.src.expert.procedural import run_expert_procedural
from backend.src.expert.scenes import Scene
from backend.src.expert.storage import (
    MAX_JSON_BYTES,
    digest,
    parse_json,
    publish_record,
    read_record,
)


def attempt_job(
    manifest: dict, row: dict, guidance_index: int, guess: int, retry: int
) -> dict:
    config = ExpertConfig.from_record(manifest["config"])
    identity = {
        "dataset_id": manifest["dataset_id"],
        "scene_index": row["index"],
        "guidance_index": guidance_index,
        "guess": guess,
        "retry": retry,
    }
    seed = int(digest(dict(identity, seed=config.dataset.seed))[:8], 16)
    return dict(
        identity,
        seed=seed,
        attempt_id=f"s{row['index']:04d}-g{guidance_index}-i{guess}-r{retry}",
        config=config.to_record(),
        observation=row["raw_observation"],
        guidance=asdict(config.guidance[guidance_index]),
        split=row["split"],
        group_id=row["group_id"],
    )


def load_attempt(path: Path, job: dict) -> dict:
    record = read_record(path)
    payload = record["payload"]
    if (
        record["payload_sha256"] != digest(payload)
        or payload["input_id"] != digest(job)
        or payload["job"] != json.loads(json.dumps(job))
    ):
        raise ValueError("Committed attempt changed or belongs to another input")
    return payload


def execute_attempt(job: dict, *, timeout: float, threads: int) -> dict:
    """A hard timeout kills the actual optimization process, not just its Future."""
    started = time.perf_counter()
    environment = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = str(threads)
    failure = {
        "status": "worker_failure",
        "solver_success": False,
        "feasible": False,
        "eligible": False,
        "validation": None,
        "states": None,
        "controls": None,
        "timestamps": None,
        "y_ref_star": None,
        "cost": None,
    }
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.run(
                [sys.executable, "-m", "backend.src.expert.worker"],
                input=json.dumps(job, allow_nan=False).encode(),
                stdout=stdout,
                stderr=stderr,
                cwd=Path(__file__).resolve().parents[3],
                env=environment,
                timeout=timeout,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            stdout.seek(0)
            if process.returncode:
                result = dict(failure, worker_exit_code=process.returncode)
            else:
                result = parse_json(stdout.read(MAX_JSON_BYTES + 1))
                if (
                    not isinstance(result, dict)
                    or not failure.keys() <= result.keys()
                    or any(
                        type(result[name]) is not bool
                        for name in ("eligible", "feasible", "solver_success")
                    )
                    or not isinstance(result["status"], str)
                ):
                    raise ValueError("Invalid worker result contract")
        except subprocess.TimeoutExpired:
            result = dict(failure, status="worker_timeout")
        except (ValueError, OSError):
            result = dict(failure, status="worker_protocol_failure")
    result["worker_seconds"] = time.perf_counter() - started
    return result


def execute_procedural_attempt(job: dict) -> dict:
    """Bounded in-process geometry avoids Python startup for every scene."""
    config = ExpertConfig.from_record(job["config"])
    result = run_expert_procedural(
        Scene.from_record(job["observation"]), Guidance(**job["guidance"]), config,
        guess=job["guess"], seed=job["seed"], retry=job["retry"],
    )
    result["worker_seconds"] = result["runtime_seconds"]
    return result


def run_expert_generation(
    manifest: dict, output: Path, *, workers: int | None = None
) -> list[dict]:
    """Caller owns dataset_lock across all stages. Resume never reruns committed attempts."""
    config = ExpertConfig.from_record(manifest["config"])
    workers = config.execution.workers if workers is None else workers
    if (
        type(workers) is not int
        or not 1 <= workers <= 16
        or workers * config.execution.threads_per_worker > (os.cpu_count() or 1)
    ):
        raise ValueError("Worker/thread budget exceeds available logical CPUs")
    started = time.perf_counter()
    results, previous = [], {}
    executed, reused = 0, 0
    sessions_path = output / "generation_sessions.json"
    sessions = read_record(sessions_path) if sessions_path.exists() else []
    for session in sessions:
        if session["state"] == "running":
            session["state"] = "interrupted"
    current_session = {
        "state": "running",
        "workers": workers,
        "threads_per_worker": config.execution.threads_per_worker,
    }
    sessions.append(current_session)
    publish_record(sessions_path, sessions)
    publish_record(
        output / "status.json",
        {"state": "running", "dataset_id": manifest["dataset_id"], "workers": workers},
    )
    print(
        "Generating offline expert trajectories: "
        f"{sum(row['status'] == 'ready' for row in manifest['scenes'])} ready scenes, "
        f"{workers} workers.",
        flush=True,
    )
    candidate_count = (
        config.planner.procedural_candidates
        if config.planner.generation_method == "procedural"
        else len(config.planner.initial_guess_offsets)
    )
    for retry in range(config.planner.retries + 1):
        jobs = []
        for row in manifest["scenes"]:
            if row["status"] != "ready":
                continue
            for guidance_index in range(len(config.guidance)):
                for guess in range(candidate_count):
                    key = (row["index"], guidance_index, guess)
                    if retry and previous[key]["result"]["eligible"]:
                        continue
                    job = attempt_job(manifest, row, guidance_index, guess, retry)
                    path = output / "attempts" / (job["attempt_id"] + ".json")
                    if path.exists():
                        reused += 1
                        payload = load_attempt(path, job)
                        results.append(payload)
                        previous[key] = payload
                    else:
                        jobs.append((key, job, path))
        total = len(jobs)
        print(
            f"Generation pass {retry + 1}/{config.planner.retries + 1}: "
            f"{total} attempts to run ({reused} receipts reused overall).",
            flush=True,
        )
        completed = 0
        report_every = max(1, total // 100)
        # Numerical optimization retains a killable subprocess. Fixed geometry
        # is bounded and runs in threads without per-attempt interpreter startup.
        local = (config.planner.generation_method == "procedural"
                 and not config.planner.optimizer_fallback)
        execute = execute_procedural_attempt if local else execute_attempt
        options = {} if local else {
            "timeout": config.planner.solver_timeout + config.execution.startup_timeout,
            "threads": config.execution.threads_per_worker,
        }
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Only the coordinator commits receipts; completion order is irrelevant.
            pending = {
                executor.submit(execute, job, **options): (key, job, path)
                for key, job, path in jobs
            }
            for future in as_completed(pending):
                key, job, path = pending[future]
                result = future.result()
                executed += 1
                completed += 1
                payload = {"input_id": digest(job), "job": job, "result": result}
                publish_record(
                    path, {"payload_sha256": digest(payload), "payload": payload}
                )
                results.append(payload)
                previous[key] = payload
                if completed == total or completed % report_every == 0:
                    # Receipts are committed per attempt; status is advisory.
                    publish_record(
                        output / "status.json",
                        {
                            "state": "running",
                            "dataset_id": manifest["dataset_id"],
                            "workers": workers,
                            "committed_attempts": len(results),
                        },
                    )
                    print(
                        f"Generation pass {retry + 1}: {completed}/{total} complete "
                        f"(latest={result['status']}, elapsed={time.perf_counter() - started:.1f}s).",
                        flush=True,
                    )
    current_session.update(
        state="finished",
        wall_seconds=time.perf_counter() - started,
        executed_attempts=executed,
        reused_attempts=reused,
    )
    publish_record(sessions_path, sessions)
    print(
        "Offline expert trajectory generation complete: "
        f"{executed} executed, {reused} reused, "
        f"{current_session['wall_seconds']:.1f}s elapsed.",
        flush=True,
    )
    return sorted(results, key=lambda item: item["job"]["attempt_id"])
