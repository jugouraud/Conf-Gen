import pytest
from hypothesis import given
from hypothesis import strategies as st

from backend.src.data.prompts import normalize_prompt_text

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text,expected",
    [
        ("  Hello, WORLD!!\n", "hello world"),
        ("Ｃａｆé", "café"),
        ("Straße", "strasse"),
        ("👩 🌈", "👩 🌈"),
        ("...", ""),
    ],
)
def test_prompt_identity_handles_unicode_and_punctuation(text, expected):
    assert normalize_prompt_text(text) == expected


@given(st.text())
def test_normalizing_prompt_identity_twice_does_not_change_it(text):
    normalized = normalize_prompt_text(text)
    assert normalize_prompt_text(normalized) == normalized
