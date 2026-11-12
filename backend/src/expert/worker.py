"""One local expert attempt. Fixed module command; bounded JSON input/output."""

import json
import sys

from threadpoolctl import threadpool_limits

from backend.src.expert.config import ExpertConfig, Guidance
from backend.src.expert.optimize import run_expert_optimization
from backend.src.expert.scenes import Scene
from backend.src.expert.storage import MAX_JSON_BYTES, json_data, parse_json


def main() -> None:
    job = parse_json(sys.stdin.buffer.read(MAX_JSON_BYTES + 1))
    config = ExpertConfig.from_record(job["config"])
    scene = Scene.from_record(job["observation"])
    guidance = Guidance(**job["guidance"])
    with threadpool_limits(limits=config.execution.threads_per_worker):
        result = run_expert_optimization(
            scene,
            guidance,
            config,
            guess=job["guess"],
            seed=job["seed"],
            retry=job["retry"],
        )
    sys.stdout.write(json.dumps(json_data(result), allow_nan=False))


if __name__ == "__main__":
    main()
