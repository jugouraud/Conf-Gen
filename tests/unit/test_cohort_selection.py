import numpy as np
import pytest

from backend.src.data.prompts import select_detected_unsafe_repair_cases

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "mode,safe_distance,unsafe_distance,selected",
    [
        ("safe_attraction", 1.01, 3, True),
        ("safe_attraction", 1, 2, False),
        ("safe_attraction_unsafe_avoidance", 1, 2, True),
        ("safe_attraction_unsafe_avoidance", 1, 2.01, False),
    ],
)
def test_category_balanced_cohort_uses_the_requested_detection_mode(
    mode, safe_distance, unsafe_distance, selected
):
    groups = {
        "unsafe_test": [
            {"id": "a", "category": "intimate_bedroom_scene"},
            {"id": "b", "category": "figure_study_nudity"},
            {"id": "c", "category": "fine_art_nude_photography"},
            {"id": "d", "category": "nude_body_positivity"},
            {"id": "e", "category": "nude_self_portrait"},
        ]
    }
    distances = {
        "safe__unsafe_test": np.full((5, 1), safe_distance),
        "unsafe__unsafe_test": np.full((5, 1), unsafe_distance),
    }
    options = {"pilot_per_category": 1, "modes": [mode], "seed": 17}
    calibration = {"safe_cutoff": 1, "unsafe_cutoff": 2}

    if selected:
        cases = select_detected_unsafe_repair_cases(
            groups, distances, calibration, options, "pilot"
        )
        assert [row["id"] for _, _, row in cases] == ["a", "b", "c", "d", "e"]
    else:
        with pytest.raises(ValueError, match="Need 1 detected-unsafe"):
            select_detected_unsafe_repair_cases(
                groups, distances, calibration, options, "pilot"
            )
