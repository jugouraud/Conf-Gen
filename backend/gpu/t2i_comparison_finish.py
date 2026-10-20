"""Finish the GPU distance/scoring stages, verify results locally, and release Colab."""

import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import zipfile

from backend.flow.comparison import safe_source_path
from backend.gpu.t2i_comparison_client import ROOT, OUTPUT, SESSION, REMOTE, colab, sha
from backend.gpu.t2i_safe_transfer import start_receiver


def record_status(values):
    path = OUTPUT / "run_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2))
    temporary.replace(path)


def read_state(path):
    for attempt in range(5):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(.2)


def wait_inputs():
    while True:
        status = read_state(OUTPUT / "status.json")
        upload_path = OUTPUT / "safe_upload_status.json"
        upload = read_state(upload_path) if upload_path.exists() else {}
        if status.get("status") == "failed":
            raise RuntimeError(status.get("error", "Unsafe extraction failed"))
        uploaded = len(upload.get("uploaded_batches", []))
        record_status({"status": "running", "phase": "extract_and_transfer",
                       "completed_features": status.get("completed_features", 0), "total_features": 4400,
                       "safe_cache_uploaded_batches": uploaded, "safe_cache_total_batches": 1500})
        if status.get("status") == "complete" and uploaded == upload.get("total_batches"):
            with tempfile.TemporaryDirectory() as directory:
                sentinel = Path(directory) / "safe-complete.json"
                try:
                    colab("download", "-s", SESSION, "/content/t2i-safe-upload-complete.json", str(sentinel))
                    if json.loads(sentinel.read_text())["batches"] == uploaded:
                        return
                except (RuntimeError, json.JSONDecodeError):
                    pass
        print(f"Waiting for extraction ({status.get('completed_features', 0)}/4400) "
              f"and safe cache upload ({len(upload.get('uploaded_batches', []))}/1500)", flush=True)
        time.sleep(30)


def launch_analysis(*, modules=("scripts.measure_t2i_distances", "scripts.score_t2i_comparison"),
                    additional_prefixes=()):
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        archive = folder / "analysis.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path in [*sorted((ROOT / "backend/flow").glob("*.py")), *sorted((ROOT / "scripts").glob("*.py"))]:
                z.write(path, path.relative_to(ROOT).as_posix())
        colab("upload", "-s", SESSION, str(archive), "/content/t2i-analysis.zip")
        worker = folder / "analysis-worker.py"
        worker.write_text('''import hashlib,json,os,subprocess,sys,time,traceback,zipfile
from pathlib import Path
root=Path('__REMOTE_ROOT__');output=root/'results'
def sha(path):
 digest=hashlib.sha256()
 with path.open('rb') as reader:
  for block in iter(lambda:reader.read(1024*1024),b''):digest.update(block)
 return digest.hexdigest()
try:
 while not (output/'safe_transfer_status.json').exists() or json.loads((output/'safe_transfer_status.json').read_text()).get('status')!='complete':time.sleep(5)
 env=os.environ.copy();env['PYTHONPATH']=str(root);env['USE_TF']='0';env['PYTHONUNBUFFERED']='1'
 for module in __EVALUATION_MODULES__:
  subprocess.run([sys.executable,'-u','-m',module,'--output',str(output)],cwd=root,env=env,check=True)
 transfer=root/'analysis_transfer';transfer.mkdir(exist_ok=True)
 paths=[p for p in output.rglob('*') if p.is_file() and ('variants' in p.relative_to(output).parts or p.name.startswith('comparison.') or p.name.startswith('two_mode_distances.') or p.name=='safe_transfer_status.json' or p.name.startswith(__ADDITIONAL_PREFIXES__))]
 manifest={p.relative_to(output).as_posix():sha(p) for p in paths}
 archive=transfer/'analysis.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for path in paths:z.write(path,path.relative_to(output).as_posix())
  z.writestr('artifact_manifest.json',json.dumps(manifest))
 parts=[]
 with archive.open('rb') as reader:
  while block:=reader.read(20*1024**2):
   part=transfer/f'analysis.part{len(parts):04d}';part.write_bytes(block)
   parts.append({'name':part.name,'sha256':sha(part),'bytes':len(block)})
 receipt={'status':'complete','archive_sha256':sha(archive),'parts':parts,'artifacts':manifest}
 (output/'analysis_result.json').write_text(json.dumps(receipt))
except BaseException:
 (output/'analysis_result.json').write_text(json.dumps({'status':'failed','error':traceback.format_exc()}))
 raise
'''.replace('__REMOTE_ROOT__', REMOTE).replace('__EVALUATION_MODULES__', repr(tuple(modules)))
            .replace('__ADDITIONAL_PREFIXES__', repr(tuple(additional_prefixes))))
        colab("upload", "-s", SESSION, str(worker), "/content/t2i-analysis-worker.py")
        launcher = folder / "launch.py"
        launcher.write_text('''import subprocess,sys,zipfile
from pathlib import Path
with zipfile.ZipFile('/content/t2i-analysis.zip') as archive:archive.extractall('__REMOTE_ROOT__')
with Path('/content/t2i-analysis.log').open('wb') as log:
 process=subprocess.Popen([sys.executable,'-u','/content/t2i-analysis-worker.py'],stdout=log,stderr=subprocess.STDOUT)
print('Distance and scoring worker PID:',process.pid)
'''.replace('__REMOTE_ROOT__', REMOTE))
        colab("exec", "-s", SESSION, "--timeout", "60", "-f", str(launcher), capture=False)


def collect_analysis():
    folder = ROOT / ".tools/t2i-analysis-transfer"
    folder.mkdir(exist_ok=True)
    result_path = folder / "analysis_result.json"
    while True:
        try:
            colab("download", "-s", SESSION, f"{REMOTE}/results/analysis_result.json", str(result_path))
            receipt = json.loads(result_path.read_text())
            break
        except (RuntimeError, json.JSONDecodeError):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "status.json"
                try:
                    colab("download", "-s", SESSION, f"{REMOTE}/results/analysis_status.json", str(path))
                    status = json.loads(path.read_text())
                    (OUTPUT / "analysis_status.json").write_text(json.dumps(status, indent=2))
                    record_status(dict(status, status="running"))
                    print(f"Analysis progress: {status}", flush=True)
                except (RuntimeError, json.JSONDecodeError):
                    pass
            time.sleep(20)
    if receipt["status"] != "complete":
        raise RuntimeError(receipt["error"])
    archive = folder / "analysis.zip"
    with archive.open("wb") as writer:
        for part in receipt["parts"]:
            name = part["name"]
            if Path(name).name != name or not name.startswith("analysis.part"):
                raise ValueError("Invalid result part name")
            local = folder / name
            for attempt in range(5):
                try:
                    colab("download", "-s", SESSION, f"{REMOTE}/analysis_transfer/{name}", str(local))
                    if local.stat().st_size != part["bytes"] or sha(local) != part["sha256"]:
                        raise ValueError("Invalid analysis part checksum")
                    break
                except (RuntimeError, ValueError):
                    if attempt == 4:
                        raise
                    time.sleep(5)
            writer.write(local.read_bytes())
    if sha(archive) != receipt["archive_sha256"]:
        raise ValueError("Analysis archive checksum mismatch")
    with zipfile.ZipFile(archive) as z:
        files = json.loads(z.read("artifact_manifest.json"))
        if files != receipt["artifacts"] or set(z.namelist()) != set(files) | {"artifact_manifest.json"}:
            raise ValueError("Analysis manifest mismatch")
        for relative, digest in files.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
                raise ValueError("Invalid analysis artifact path")
            content = z.read(relative)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("Invalid analysis artifact checksum")
            target = OUTPUT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha(target) != digest:
                raise ValueError(f"Conflicting local analysis result: {relative}")
            target.write_bytes(content)
    colab("download", "-s", SESSION, "/content/t2i-analysis.log", str(OUTPUT / "analysis_worker.log"))
    verify_safe_reuse()
    (OUTPUT / "analysis_status.json").write_text(json.dumps({"status": "complete", "phase": "comparison_complete",
                                                           "verified_artifacts": len(files)}, indent=2))
    colab("stop", "-s", SESSION)
    record_status({"status": "complete", "phase": "comparison_complete", "methods": 5, "modes": 2,
                   "unsafe_anchors": 1000, "unsafe_calibration": 200, "unsafe_test": 1000,
                   "verified_artifacts": len(files), "gpu_session_stopped": True})
    print("All ten method/mode results verified locally; A100 session stopped.", flush=True)


def verify_safe_reuse():
    comparison = json.loads((OUTPUT / "comparison.json").read_text())
    manifest = comparison["experiment"]
    source = safe_source_path(ROOT / manifest["settings"]["paths"]["source"])
    if sha(source / "experiment.json") != manifest["source_experiment_sha256"]:
        raise ValueError("The original safe experiment changed during this run")
    checked = {}
    for key, row in comparison["methods"].items():
        if row["mode"] != "safe_attraction":
            continue
        original = source / "variants" / f"{row['space']}_{row['method']}" / "scores"
        baseline = json.loads((original / "report.json").read_text())
        if (not math.isclose(row["calibration"]["safe_cutoff"], baseline["threshold"], rel_tol=1e-12, abs_tol=1e-12)
                or row["safe_false_positive_rate"] != baseline["safe_false_positive_rate"]):
            raise ValueError("Reused safe scores differ from the original baseline")
        checked[key] = {"safe_cutoff": row["calibration"]["safe_cutoff"],
                        "safe_false_positive_rate": row["safe_false_positive_rate"]}
    if len(checked) != 5 or len(comparison["methods"]) != 10:
        raise ValueError("Expected exactly five methods in two modes")
    (OUTPUT / "safe_reuse_validation.json").write_text(json.dumps({"status": "verified", "methods": checked}, indent=2))


if __name__ == "__main__":
    try:
        wait_inputs()
        start_receiver()
        launch_analysis()
        collect_analysis()
    except BaseException as error:
        record_status({"status": "failed", "error": str(error), "gpu_session": SESSION})
        raise
