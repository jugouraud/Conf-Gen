# Conf-Gen
Inner-embedding safety mechanisms in large models

## Structure

- `backend/`: database creation, model storage, embedding extraction, harmfulness scoring, data filling, and visualization state.
- `frontend/`: NiceGUI page layout and application launcher.

## Commands

Run the commands as modules from the repository root:

```powershell
# Open the NiceGUI visualization in a browser.
.\.venv\Scripts\python.exe -m frontend.visualize_database_embeddings --database data\fairness_data.sqlite3

# Fill one or more database output fields.
.\.venv\Scripts\python.exe -m backend.filling_database --prompt-harmfulness
```

## Colab GPU embeddings

The embedding job can run on a Google Colab GPU through the official
[Google Colab CLI](https://github.com/googlecolab/google-colab-cli). The CLI
currently supports Linux and macOS; on Windows, run it through WSL.

Install WSL from an elevated PowerShell terminal if it is not already present,
then restart Windows:

```powershell
wsl --install -d Ubuntu
```

Inside WSL, install the pinned CLI version and run the GPU job from this
repository:

```bash
bash scripts/setup_colab_cli.sh
python3 scripts/colab_embeddings.py --gpu T4
```

From PowerShell, the second command can instead be launched through the wrapper:

```powershell
.\scripts\run_colab_embeddings.ps1 --gpu T4
```

The first run opens Google authentication. The command snapshots the SQLite
database and referenced images, requests a GPU runtime, verifies CUDA, fills all
inner embeddings, downloads `data/fairness_data.embedded.sqlite3`, and releases
the runtime even if the job fails. It never overwrites the source database by
default. Use `--output PATH`, `--gpu L4`, `--high-mem`, or `--overwrite` when
needed. GPU availability and some accelerator types depend on the Colab account.
