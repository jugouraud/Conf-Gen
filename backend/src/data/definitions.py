"""Shared prompt partition names and label conventions."""

SPLITS = ("safe_reference", "safe_validation", "safe_test", "unsafe_test")

SAFE_INPUT = {"safe_reference", "safe_calibration", "safe_test"}

SOURCE_SPLIT_ORDER = {"safe_reference": 0, "safe_calibration": 1, "safe_test": 2}

SAFE_TARGET_SPLITS = {
    "safe_reference": "safe_reference",
    "safe_calibration": "safe_validation",
    "safe_test": "safe_test",
}

UNSAFE_SPLITS = ("unsafe_reference", "unsafe_validation", "unsafe_test")

SAFE_SPLITS = ("safe_reference", "safe_validation", "safe_test")

SPLIT = "additional_safe_test"
