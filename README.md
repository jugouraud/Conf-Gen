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
