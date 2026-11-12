"""One local expert attempt. Fixed module command; bounded JSON input/output."""

import json
import sys

from threadpoolctl import threadpool_limits

from backend.src.expert.config import ExpertConfig, Guidance
from backend.src.expert.optimize import run_expert_optimization
from backend.src.expert.procedural import run_expert_procedural
from backend.src.expert.scenes import Scene
from backend.src.expert.storage import MAX_JSON_BYTES, json_data, parse_json


def main() -> None:
    job = parse_json(sys.stdin.buffer.read(MAX_JSON_BYTES + 1))
    config = ExpertConfig.from_record(job["config"])
    scene = Scene.from_record(job["observation"])
    guidance = Guidance(**job["guidance"])
    with threadpool_limits(limits=config.execution.threads_per_worker):
        if config.planner.generation_method == "procedural":
            result = run_expert_procedural(
                scene, guidance, config, guess=job["guess"],
                seed=job["seed"], retry=job["retry"],
            )
            if not result["eligible"] and config.planner.optimizer_fallback and (
                result["status"] != "invalid_scene_or_guidance"
            ):
                result = run_expert_optimization(
                    scene, guidance, config,
                    guess=job["guess"] % len(config.planner.initial_guess_offsets),
                    seed=job["seed"], retry=job["retry"],
                )
                result["method"] = "optimization_fallback"
        else:
            result = run_expert_optimization(
                scene, guidance, config, guess=job["guess"],
                seed=job["seed"], retry=job["retry"],
            )
            result["method"] = "optimization"
    sys.stdout.write(json.dumps(json_data(result), allow_nan=False))


if __name__ == "__main__":
    main()
