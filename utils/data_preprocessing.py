"""Load prompt-image pairs from the local evaluation data."""

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
DEFAULT_RECORDS_FILE = "hf_test_fairness_real.json"


@dataclass(frozen=True, slots=True)
class PromptImagePair:
    """The prompt and local image belonging to one dataset record."""

    prompt: str
    image: Path


def load_prompt_image_pair(
    record_id: int,
    *,
    data_directory: str | Path = DEFAULT_DATA_DIRECTORY,
    records_file: str = DEFAULT_RECORDS_FILE,
) -> PromptImagePair:
    """Return the prompt and image for ``record_id`` from the local data set."""
    if not isinstance(record_id, int) or isinstance(record_id, bool):
        raise TypeError("record_id must be an integer.")

    data_root = Path(data_directory)
    record = _find_record(data_root / records_file, record_id)
    prompt = _read_prompt(record, record_id)
    image = _resolve_image_path(data_root, record, record_id)
    return PromptImagePair(prompt=prompt, image=image)


def _find_record(records_path: Path, record_id: int) -> dict:
    """Return the JSON record with ``record_id`` or raise a clear error."""
    try:
        records = json.loads(records_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Data records file not found: {records_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Data records file is not valid JSON: {records_path}") from error

    if not isinstance(records, list):
        raise ValueError("Data records must be a JSON list.")
    for record in records:
        if isinstance(record, dict) and record.get("id") == record_id:
            return record
    raise KeyError(f"No data record found for id {record_id}.")


def _read_prompt(record: dict, record_id: int) -> str:
    """Read and validate the record caption used as the prompt."""
    prompt = record.get("caption")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Record {record_id} has no usable caption.")
    return prompt


def _resolve_image_path(data_root: Path, record: dict, record_id: int) -> Path:
    """Resolve the record's image reference to its local image file."""
    reference = record.get("image")
    if isinstance(reference, list) and len(reference) == 1:
        reference = reference[0]
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"Record {record_id} must contain one image path.")

    image_path = data_root / Path(reference).name
    if not image_path.is_file():
        raise FileNotFoundError(f"Image for record {record_id} not found: {image_path}")
    return image_path
