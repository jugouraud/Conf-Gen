# Reusable Colab GPU session

Install the Colab CLI in WSL with `bash scripts/setup_colab.sh`, then authenticate with `colab --auth oauth2 sessions`. The generic runner can create a named session, upload any local files, run shell commands in `/content`, and download results. It leaves the session running unless `--stop` is supplied.

From PowerShell:

```powershell
.\scripts\gpu.ps1 --session shared-gpu --create --gpu T4 `
  --upload .\jobs\train.py /content/train.py `
  --run "python /content/train.py" `
  --download /content/model.pt .\results\model.pt

.\scripts\gpu.ps1 --session shared-gpu `
  --upload .\jobs\analyze.py /content/analyze.py `
  --run "python /content/analyze.py" `
  --download /content/report.csv .\results\report.csv

.\scripts\gpu.ps1 --session shared-gpu --stop
```

PowerShell local upload and download paths are converted for WSL. In WSL, use the same arguments with `bash scripts/gpu.sh`. Repeat `--upload LOCAL REMOTE`, `--run "COMMAND"`, and `--download REMOTE LOCAL` as needed. Remote file paths must be absolute POSIX paths. Commands run in order, each in `/content`; a failing command stops the job and is reported locally. Each command starts a new shell, so put stateful shell steps in one command or an uploaded script. `--timeout SECONDS` controls the execution limit. `--auth adc` and `--high-mem` are also supported.

To use the same session for the safe-flow experiment, start its first stage with `--session shared-gpu --keep-session`. Later stages can use `--session shared-gpu --reuse-session`. Both modes keep the runtime available for other jobs. The ordinary safe-flow command still creates and stops its own session.
