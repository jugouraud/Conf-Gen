"""Stage a prepared repair bundle and its verified recovered field cache."""

from pathlib import Path

from backend.runtime.colab import ColabConnection
from backend.src.common.archives import resolve_contained_artifact_path
from backend.src.common.storage import calculate_file_sha256, read_json, write_json


class RepairColabConnection(ColabConnection):
    """Repair uses an explicit package path and optional multipart cache recovery."""

    def stage(self) -> dict:
        self.verify_hardware()
        validation = read_json(self.output / "validation.json")
        package = Path(validation["package_path"])
        if calculate_file_sha256(package) != validation["package_sha256"]:
            raise ValueError("Prepared repair package changed")
        metadata_path = self.output / "gpu/reuse_upload.json"
        uploads = []
        if metadata_path.exists():
            upload = read_json(metadata_path)
            metadata = read_json(Path(upload["metadata"]))
            expected = read_json(self.output / "experiment.json")["reuse"][
                "archive_sha256"
            ]
            if metadata["sha256"] != expected:
                raise ValueError("Recovered archive identity changed")
            for part in metadata["parts"]:
                path = resolve_contained_artifact_path(
                    upload["parts_directory"], part["name"]
                )
                if calculate_file_sha256(path) != part["sha256"]:
                    raise ValueError("Recovered archive part changed")
                uploads.append((path, part["sha256"]))
        remote_package = self.root + "-inputs.zip"
        self.invoke(
            "upload", "-s", self.session, self.local_argument(package), remote_package
        )
        # Only fixed source is sent to exec. The bundle is unpacked by the shared
        # archive validator, which checks paths and all checksums before writes.
        self.execute(f"""import hashlib,json,zipfile
from pathlib import Path
package=Path({remote_package!r});root=Path({self.root!r})
if hashlib.file_digest(package.open('rb'),'sha256').hexdigest()!={validation["package_sha256"]!r}:raise ValueError('Upload checksum mismatch')
with zipfile.ZipFile(package) as bundle:
 names=bundle.namelist();manifest=json.loads(bundle.read('artifact_manifest.json'))
 if len(names)!=len(set(names)) or set(names)!=set(manifest)|{{'artifact_manifest.json'}}:raise ValueError('Unexpected package contents')
 for name,digest in manifest.items():
  target=(root/name).resolve()
  if not target.is_relative_to(root.resolve()) or '\\\\' in name or ':' in name:raise ValueError('Package path escapes root')
  if hashlib.sha256(bundle.read(name)).hexdigest()!=digest:raise ValueError('Package checksum mismatch')
  if target.exists() and hashlib.file_digest(target.open('rb'),'sha256').hexdigest()!=digest:raise ValueError('Existing staged bytes differ; use a fresh root')
 for name in manifest:
  target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():target.write_bytes(bundle.read(name))
""")
        for path, _digest in uploads:
            self.invoke(
                "upload",
                "-s",
                self.session,
                self.local_argument(path),
                self.root + "-" + path.name,
            )
        parts = [(self.root + "-" + path.name, digest) for path, digest in uploads]
        source = f"""import hashlib,json,sys,tempfile
from pathlib import Path
root=Path({self.root!r});sys.path.insert(0,str(root))
from backend.src.common.archives import extract_verified_archive
parts={parts!r}
if parts:
 with tempfile.TemporaryDirectory(dir=root) as scratch:
  archive=Path(scratch)/'recovered.zip'
  with archive.open('wb') as writer:
   for name,digest in parts:
    path=Path(name)
    with path.open('rb') as reader:
     if hashlib.file_digest(reader,'sha256').hexdigest()!=digest:raise ValueError('Cache part checksum mismatch')
     reader.seek(0)
     while block:=reader.read(8*1024**2):writer.write(block)
  manifest=json.loads((root/'results/experiment.json').read_text())
  with archive.open('rb') as reader:
   if hashlib.file_digest(reader,'sha256').hexdigest()!=manifest['reuse']['archive_sha256']:raise ValueError('Cache archive checksum mismatch')
  extract_verified_archive(archive,root/'results',expected_hashes=manifest['reuse']['files'])
print('CACHE_STAGE='+json.dumps({{'status':'complete','files':len(parts)}}))
"""
        response = self.execute(source)
        self.stage_model_auth()
        write_json(
            self.output / "progress.json", {"status": "starting", "phase": "staged"}
        )
        for line in response.splitlines():
            if line.startswith("CACHE_STAGE="):
                import json

                return json.loads(line.removeprefix("CACHE_STAGE="))
        raise RuntimeError("No cache staging receipt returned")
