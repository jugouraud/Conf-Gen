import pytest

from backend.src.detection.calibration import calculate_detection_metrics

pytestmark = pytest.mark.unit


def test_detection_metrics_separate_false_alarms_from_unsafe_detection():
    result = calculate_detection_metrics(
        labels=[False, False, True, True],
        flags=[False, True, True, True],
        scores=[0.1, 0.3, 0.8, 0.9],
    )
    assert result == {
        "safe_test_n": 2,
        "unsafe_test_n": 2,
        "safe_false_positive_rate": 0.5,
        "unsafe_flagged_rate": 1.0,
        "auroc_prompt_labels": 1.0,
        "auprc_prompt_labels": 1.0,
    }
