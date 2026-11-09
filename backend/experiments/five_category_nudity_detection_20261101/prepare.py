"""Run the five-category nudity detection experiment."""

from collections import Counter
from pathlib import Path

from backend.experiments.five_category_nudity_detection_20261101.definition import (
    CATEGORIES,
)
from backend.src.common.storage import (
    calculate_source_fingerprints,
    read_json,
    read_toml_config,
    relocate_path,
    write_json,
)
from backend.src.data.paired import read_paired_detection_prompts
from backend.src.detection.contracts import IMPLEMENTATION_FILES, SPACE_METHODS
from backend.src.detection.prepare import run_detection_preparation


def run_nudity_detection_preparation(config: Path, output: Path) -> dict:
    """Freeze 1,000 paired inputs, source identities and CPU-only resource estimates."""
    settings = read_toml_config(config)
    settings["paths"]["output"] = str(Path(output).resolve())
    groups, audit = read_paired_detection_prompts(
        relocate_path(settings["paths"]["dataset"]),
        prompt_columns=settings["dataset"]["prompt_columns"],
    )
    if audit["pairs"] != 1000:
        raise ValueError("This experiment requires exactly 1,000 prompt pairs")
    for label in ("safe", "unsafe"):
        for split, count in (("reference", 140), ("validation", 20), ("test", 40)):
            counts = Counter(row["category"] for row in groups[f"{label}_{split}"])
            if counts != {category: count for category in CATEGORIES}:
                raise ValueError(
                    "Nudity detection requires five categories with 140/20/40 pairs each"
                )
    run_detection_preparation(settings, output, mode="paired")
    manifest = read_json(output / "experiment.json")
    manifest.update(
        implementation=calculate_source_fingerprints(IMPLEMENTATION_FILES),
        experiment_implementation=calculate_source_fingerprints(
            f"backend/experiments/five_category_nudity_detection_20261101/{name}.py"
            for name in ("definition", "prepare", "package", "pipeline")
        ),
        variants={space: list(methods) for space, methods in SPACE_METHODS.items()},
        probe_policy="Fresh safe-training trajectories; paired Gaussian source states from their seeds",
        transfer_policy="Compact distances and reports; raw fields and embeddings retained only on GPU host",
        scope="Five-category nudity prompt-label detection; no image generation or repair",
    )
    write_json(output / "experiment.json", manifest)
    query_count = sum(
        len(groups[split]) for split in groups if not split.endswith("reference")
    )
    coordinates = (
        16 * (settings["model"]["height"] // 8) * (settings["model"]["width"] // 8)
    )
    probes = settings["probes"]["time_points"] * settings["probes"]["trajectories"]
    write_json(
        output / "resource_plan.json",
        {
            "pairs": audit["pairs"],
            "counts": manifest["counts"],
            "trajectory_query_cache_gib": query_count
            * probes
            * coordinates
            * 4
            / 1024**3,
            "trajectory_field_checkpoints_gib": 2000
            * probes
            * coordinates
            * 4
            / 1024**3,
            "source_field_checkpoints_gib": 2000
            * settings["probes"]["trajectories"]
            * coordinates
            * 4
            / 1024**3,
            "recommended_gpu_vram_gib": 15,
            "recommended_free_host_ram_gib": 32,
            "recommended_free_disk_gib": 80,
            "gpu_started": False,
            "notes": "Colab T4 high-RAM; inference and scoring use GPU in separate phases. FP64 arithmetic, FP16 inference. Allow max-deviation scratch alongside raw checkpoints and model-cache disk space.",
        },
    )
    return manifest["counts"]
