"""Drive paired detection through the authenticated Colab session CLI."""

import json
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from backend.src.common.archives import extract_verified_archive
from backend.src.common.storage import calculate_file_sha256, read_json, write_json

COMPACT_PATTERNS = (
    "experiment.json",
    "resource_plan.json",
    "banks.provenance.json",
    "status.json",
    "trajectory/splits/*.jsonl",
    "*/distances.npz",
    "*/distances.provenance.json",
    "optimization/*.json",
    "scores/**/*.json",
    "scores/**/*.csv",
    "comparison.*",
    "max_deviation/experiment.json",
    "max_deviation/status.json",
    "max_deviation/comparison.*",
    "max_deviation/*/experiment.json",
    "max_deviation/*/distances.npz",
    "max_deviation/*/distances.provenance.json",
    "max_deviation/*/scores/**/*.json",
    "max_deviation/*/scores/**/*.csv",
    "gpu_worker_status.json",
    "gpu_worker.log",
    "failure.json",
)


class ColabConnection:
    """One named session; no tensors pass through remote exec or monitoring."""

    def __init__(self, output):
        self.output = Path(output).resolve()
        self.settings = read_json(self.output / "experiment.json")["config"]["colab"]
        self.command = list(self.settings["command"])
        self.session = self.settings["session"]
        self.root = self.settings["remote_root"]
        path = PurePosixPath(self.root)
        if not path.is_absolute() or ".." in path.parts or str(path) == "/":
            raise ValueError("Colab requires a dedicated absolute remote root")
        self.gpu = self.settings["gpu"]
        self.fallback_gpu = self.settings.get("fallback_gpu")
        if self.gpu not in {"T4", "A100"} or self.fallback_gpu not in {None, "T4"}:
            raise ValueError("Unsupported Colab GPU configuration")

    def local_argument(self, path):
        value = str(Path(path).resolve())
        if Path(self.command[0]).stem.lower() == "wsl" and PureWindowsPath(value).drive:
            windows = PureWindowsPath(value)
            return f"/mnt/{windows.drive[0].lower()}/" + "/".join(windows.parts[1:])
        return value

    def invoke(self, *arguments, timeout=90):
        result = subprocess.run(
            [*self.command, "--auth", self.settings.get("auth", "oauth2"), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        safe = re.sub(r"https?://\S+", "[runtime URL]", result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(
                f"Colab {arguments[0]} failed ({result.returncode}): {safe[-2500:]}"
            )
        return safe

    def execute(self, source):
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "operation.py"
            script.write_text(source, encoding="utf-8")
            return self.invoke(
                "exec",
                "-s",
                self.session,
                "-f",
                self.local_argument(script),
                "--timeout",
                "45",
                timeout=65,
            )

    def preflight(self):
        sessions = self.invoke("sessions")
        lines = [line for line in sessions.splitlines() if f"[{self.session}]" in line]
        allowed = (
            self.gpu,
            *(() if self.fallback_gpu is None else (self.fallback_gpu,)),
        )
        if lines and not all(
            any(f"Hardware: {gpu}" in line for gpu in allowed) for line in lines
        ):
            raise RuntimeError(
                f"Named session already exists on hardware other than {'/'.join(allowed)}"
            )
        receipt = {
            "authenticated": True,
            "command": self.command,
            "session": self.session,
            "session_exists": bool(lines),
            "gpu": self.gpu,
            "remote_root": self.root,
            "gpu_started_by_preflight": False,
        }
        write_json(self.output / "gpu/colab_preflight.json", receipt)
        return receipt

    def create(self):
        requested = (
            self.gpu,
            *(() if self.fallback_gpu is None else (self.fallback_gpu,)),
        )
        error = None
        for gpu in requested:
            try:
                return self.invoke(
                    "new", "-s", self.session, "--gpu", gpu, "--high-mem", timeout=150
                )
            except RuntimeError as exc:
                error = exc
        raise error

    def verify_hardware(self):
        allowed = repr(
            (self.gpu, *(() if self.fallback_gpu is None else (self.fallback_gpu,)))
        )
        expected = f"Expected exactly one {'/'.join((self.gpu, *(() if self.fallback_gpu is None else (self.fallback_gpu,))))}"
        return self.execute(f"""import json,shutil,subprocess
from pathlib import Path
names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
if len(names)!=1 or not any(gpu in names[0] for gpu in {allowed}):raise RuntimeError({expected!r})
print(json.dumps({{'gpu':names[0],'disk_free_gib':shutil.disk_usage('/content').free/1024**3}}))
""")

    def stage(self):
        state = self.preflight()
        if not state["session_exists"]:
            raise RuntimeError(
                "Named session does not exist; use colab-launch or create it explicitly"
            )
        self.verify_hardware()
        archive = self.output.parent / "gpu-inputs.zip"
        if not archive.exists():
            raise ValueError("Package the current experiment before Colab staging")
        target = self.root + "-inputs.zip"
        self.invoke("upload", "-s", self.session, self.local_argument(archive), target)
        result = self.execute(f"""import hashlib,json,zipfile
from pathlib import Path
archive=Path({target!r});root=Path({self.root!r})
if hashlib.sha256(archive.read_bytes()).hexdigest()!={calculate_file_sha256(archive)!r}:raise ValueError('Upload checksum mismatch')
with zipfile.ZipFile(archive) as bundle:
 manifest=json.loads(bundle.read('artifact_manifest.json'))
 if len(bundle.namelist())!=len(manifest)+1 or set(bundle.namelist())!=set(manifest)|{{'artifact_manifest.json'}}:raise ValueError('Unexpected staged files')
 for name,digest in manifest.items():
  destination=(root/name).resolve()
  if not destination.is_relative_to(root.resolve()):raise ValueError('Staged path escapes root')
  data=bundle.read(name)
  if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Staged checksum mismatch')
  if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:raise ValueError('Existing staged bytes differ; use a fresh session/root')
 for name in manifest:
  destination=root/name;destination.parent.mkdir(parents=True,exist_ok=True)
  if not destination.exists():destination.write_bytes(bundle.read(name))
print(json.dumps({{'status':'staged','files':len(manifest),'root':str(root)}}))
""")
        self.stage_model_auth()
        return result

    def stage_model_auth(self):
        """Forward an existing model credential separately from source/results."""
        target = self.settings.get("hf_token_path")
        if not target:
            return
        from huggingface_hub import get_token

        token = get_token()
        if not token and Path(self.command[0]).stem.lower() == "wsl":
            prefix = self.command[: self.command.index("--") + 1]
            source = (
                "import os; from pathlib import Path; "
                "p=Path(os.environ.get('HF_TOKEN_PATH',str(Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'token'))); "
                "print(os.environ.get('HF_TOKEN') or (p.read_text().strip() if p.is_file() else ''))"
            )
            credential = subprocess.run(
                [*prefix, "python3", "-c", source],
                capture_output=True,
                text=True,
                timeout=20,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if credential.returncode == 0:
                token = credential.stdout.strip()
        if not token:
            raise RuntimeError(
                "The existing Hugging Face model credential is unavailable"
            )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model-token"
            path.write_text(token, encoding="utf-8")
            self.invoke("upload", "-s", self.session, self.local_argument(path), target)
        self.execute(
            f"import os; os.chmod({target!r}, 0o600); print('Model authentication staged')"
        )

    def launch(self):
        state = self.preflight()
        if not state["session_exists"]:
            arguments = ["new", "-s", self.session, "--gpu", "T4"]
            if self.settings.get("high_mem", True):
                arguments.append("--high-mem")
            self.invoke(*arguments, timeout=150)
        self.stage()
        # Dependency installation and inference run in a detached process. The
        # CLI only starts it, then status/collection use separate short requests.
        worker = """import importlib.metadata,json,os,subprocess,sys,traceback
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
   if requirement.name=='torch':raise RuntimeError('Colab native torch is missing/incompatible; refusing a CUDA stack replacement')
   missing.append(spec)
 if missing:subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
 update(status='running',phase='inference_and_gpu_scoring')
 code=subprocess.call([sys.executable,'-u','-m','backend.runtime.detection','--output',str(output)],cwd=root)
 update(status='complete' if code==0 else 'failed',exit_code=code)
except BaseException:
 update(status='failed',error=traceback.format_exc());raise
"""
        return self.execute(f"""import json,os,subprocess,sys
from pathlib import Path
root=Path({self.root!r});results=root/'results';pid_path=results/'worker.pid.json'
if pid_path.exists():
 old=json.loads(pid_path.read_text());process=Path('/proc')/str(old['pid'])
 if process.exists() and str(root).encode() in (process/'cmdline').read_bytes():raise RuntimeError('This experiment already has a live worker')
worker=root/'colab_worker.py';worker.write_text({worker!r})
env=os.environ.copy();env.update(PYTHONPATH=str(root),PYTHONUNBUFFERED='1',USE_TF='0',MPLBACKEND='Agg')
env.update({ ({"HF_TOKEN_PATH": self.settings["hf_token_path"]} if self.settings.get("hf_token_path") else {})!r})
with (results/'gpu_worker.log').open('a') as log:
 process=subprocess.Popen([sys.executable,'-u',str(worker)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pid_path.write_text(json.dumps({{'pid':process.pid}}))
print(json.dumps({{'status':'launched','pid':process.pid,'root':str(root)}}))
""")

    def status(self):
        return self.execute(f"""import json
from pathlib import Path
root=Path({self.root!r})/'results'
print(json.dumps({{name:json.loads((root/name).read_text()) for name in ('gpu_worker_status.json','status.json') if (root/name).exists()}}))
""")

    def snapshot(self):
        """Read only small worker/progress records; never transfer field tensors."""
        response = self.execute(f"""import json,os,re
from pathlib import Path
root=Path({self.root!r})/'results'
names=('gpu_worker_status.json','status.json','failure.json',
       'max_deviation/status.json','max_deviation/trajectory/status.json',
       'max_deviation/source_slice/status.json')
records={{}}
for name in names:
 path=root/name
 if path.exists():
  try:records[name]=json.loads(path.read_text())
  except (OSError,ValueError):pass
alive=None;pid=None
if (root/'worker.pid.json').exists():
 pid=json.loads((root/'worker.pid.json').read_text())['pid']
 try:os.kill(pid,0);alive=True
 except ProcessLookupError:alive=False
payload={{'records':records,'worker_pid':pid,'worker_alive':alive,
         'results_present':{{'rms':(root/'comparison.json').exists(),
                            'max':(root/'max_deviation/comparison.json').exists()}}}}
if alive is False or records.get('gpu_worker_status.json',{{}}).get('status')=='failed':
 path=root/'gpu_worker.log'
 if path.exists():
  with path.open('rb') as stream:
   stream.seek(max(0,path.stat().st_size-4000));payload['log_tail']=stream.read().decode(errors='replace')
encoded=json.dumps(payload)
encoded=re.sub(r'https?://[^\\s"\\\\]+','[runtime URL]',encoded)
print('PAIRED_SNAPSHOT='+encoded)
""")
        for line in response.splitlines():
            if "PAIRED_SNAPSHOT=" in line:
                return json.loads(line.split("PAIRED_SNAPSHOT=", 1)[1])
        raise RuntimeError("Colab returned no structured progress snapshot")

    def collect(self):
        self.execute(f"""import hashlib,json,zipfile
from pathlib import Path
root=Path({self.root!r});results=root/'results'
status=results/'gpu_worker_status.json'
if not status.exists() or json.loads(status.read_text()).get('status') not in ('complete','failed'):raise RuntimeError('Wait for a terminal worker state before collection')
files={{p.relative_to(results).as_posix():p for pattern in {COMPACT_PATTERNS!r} for p in results.glob(pattern) if p.is_file()}}
manifest={{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}}
archive=root/'compact-results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as bundle:
 for name,path in files.items():bundle.write(path,name)
 bundle.writestr('artifact_manifest.json',json.dumps(manifest))
parts=[]
with archive.open('rb') as reader:
 while block:=reader.read(24*1024**2):
  part=root/f'compact.part{{len(parts):04d}}';part.write_bytes(block)
  parts.append({{'name':part.name,'sha256':hashlib.sha256(block).hexdigest()}})
(root/'collection.json').write_text(json.dumps({{'parts':parts,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}}))
""")
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            receipt = folder / "collection.json"
            self.invoke(
                "download",
                "-s",
                self.session,
                self.root + "/collection.json",
                self.local_argument(receipt),
            )
            metadata = read_json(receipt)
            archive = folder / "compact-results.zip"
            with archive.open("wb") as writer:
                for part in metadata["parts"]:
                    if PurePosixPath(part["name"]).name != part["name"]:
                        raise ValueError("Invalid collection part")
                    local = folder / part["name"]
                    self.invoke(
                        "download",
                        "-s",
                        self.session,
                        self.root + "/" + part["name"],
                        self.local_argument(local),
                    )
                    if calculate_file_sha256(local) != part["sha256"]:
                        raise ValueError("Collection part checksum mismatch")
                    writer.write(local.read_bytes())
            if calculate_file_sha256(archive) != metadata["sha256"]:
                raise ValueError("Collection archive checksum mismatch")
            manifest = extract_verified_archive(
                archive,
                self.output,
                replace_names=(
                    "status.json",
                    "gpu_worker_status.json",
                    "gpu_worker.log",
                    "failure.json",
                ),
            )
        write_json(
            self.output / "gpu/colab_collection.json",
            {"files": manifest, "session": self.session},
        )
        return {"collected_files": len(manifest), "raw_fields_transferred": 0}


def run_colab_operation(output, operation):
    connection = ColabConnection(output)
    operations = {
        "preflight": connection.preflight,
        "stage": connection.stage,
        "launch": connection.launch,
        "status": connection.status,
        "collect": connection.collect,
    }
    return operations[operation]()
