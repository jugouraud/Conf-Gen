"""Persistent CLIP prompt encodings shared by live checks and batch reports."""

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from backend.paths import VALIDATION_ROOT

DEFAULT_VALIDATION_DATABASE_PATH = VALIDATION_ROOT / "validation.sqlite3"


class PromptEncodingStore:
    """Store raw EOS vectors and content-token clouds, keyed by encoder and prompt."""

    def __init__(self, path: str | Path = DEFAULT_VALIDATION_DATABASE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS prompt_encodings (
                    encoder_key TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    cloud BLOB NOT NULL,
                    cloud_rows INTEGER NOT NULL,
                    dimension INTEGER NOT NULL,
                    PRIMARY KEY (encoder_key, prompt)
                )
            """)
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def read_many(
        self, encoder_key: str, prompts: list[str], dimension: int,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Ignore malformed or dimension-mismatched rows so they can be rebuilt."""
        found = {}
        with closing(self._connect()) as connection:
            for start in range(0, len(prompts), 400):
                chunk = prompts[start:start + 400]
                if not chunk:
                    continue
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT prompt, vector, cloud, cloud_rows, dimension FROM prompt_encodings "
                    f"WHERE encoder_key = ? AND prompt IN ({placeholders})",
                    (encoder_key, *chunk),
                )
                for prompt, vector_blob, cloud_blob, cloud_rows, stored_dimension in rows:
                    if (stored_dimension != dimension or cloud_rows < 1 or
                        len(vector_blob) != dimension * 4 or
                        len(cloud_blob) != cloud_rows * dimension * 4):
                        continue
                    vector = np.frombuffer(vector_blob, dtype="<f4").copy()
                    cloud = np.frombuffer(cloud_blob, dtype="<f4").copy().reshape(cloud_rows, dimension)
                    if np.isfinite(vector).all() and np.isfinite(cloud).all():
                        found[prompt] = vector, cloud
        return found

    def write_many(
        self, encoder_key: str, values: dict[str, tuple[np.ndarray, np.ndarray]], dimension: int,
    ) -> None:
        rows = []
        for prompt, (vector, cloud) in values.items():
            vector = np.asarray(vector, dtype="<f4")
            cloud = np.asarray(cloud, dtype="<f4")
            if (vector.shape != (dimension,) or cloud.ndim != 2 or
                cloud.shape[1] != dimension or cloud.shape[0] < 1 or
                not np.isfinite(vector).all() or not np.isfinite(cloud).all()):
                raise ValueError(f"Invalid CLIP encoding for prompt: {prompt[:80]}")
            rows.append((encoder_key, prompt, vector.tobytes(), cloud.tobytes(), cloud.shape[0], dimension))
        if rows:
            with closing(self._connect()) as connection:
                with connection:
                    connection.executemany(
                        "INSERT OR REPLACE INTO prompt_encodings "
                        "(encoder_key, prompt, vector, cloud, cloud_rows, dimension) VALUES (?, ?, ?, ?, ?, ?)",
                        rows,
                    )
