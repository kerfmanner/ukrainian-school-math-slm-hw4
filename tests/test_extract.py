from __future__ import annotations

import pytest

from math_solver.extract import (
    ExtractionMethod,
    clean_latex,
    extract_answer,
    last_boxed_content,
    marker_segments,
)


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("204 × 87 = 17748\nВідповідь: 17748", "17748"),
        ("Відповідь: 20,7", "20.7"),
        ("Відповідь: 3.50.", "3.5"),
        ("Відповідь: 12 см.", "12"),
        ("Відповідь: 988 300 г", "988300"),
        ("Відповідь: 0,001012", "0.001012"),
        ("Відповідь: 207/10", "207/10"),
        ("Відповідь: 18/20", "9/10"),
        ("Відповідь: 20 7/10", "207/10"),
        ("відповідь: 5", "5"),
        ("ВІДПОВІДЬ: 5", "5"),
        ("**Відповідь:** 42", "42"),
        ("**Відповідь**: 42", "42"),
        ("Відповідь — 42", "42"),
        ("Відповідь: $17{,}7$", "17.7"),
        (r"Відповідь: \(\frac{9}{10}\)", "9/10"),
        (r"Відповідь: $\boxed{15}$", "15"),
        (r"Відповідь: $\boxed{\dfrac{3}{4}}$", "3/4"),
        (r"Відповідь: $988\,300$", "988300"),
        (r"Відповідь: $12\text{ см}$", "12"),
        (r"Відповідь: $50\%$", "50"),
        ("Відповідь: 204 × 87 = 17748", "17748"),
        ("Відповідь: x = 7", "7"),
        ("Відповідь: `25`", "25"),
        ("Відповідь:\n\n  36  \n", "36"),
    ],
)
def test_extract_answer_from_marker(completion: str, expected: str) -> None:
    result = extract_answer(completion)
    assert result.answer == expected
    assert result.method is ExtractionMethod.MARKER


def test_repeated_markers_use_last_parseable() -> None:
    completion = "Відповідь: 10\nПеревірка показала помилку.\nВідповідь: 12"
    assert extract_answer(completion).answer == "12"


def test_repeated_markers_skip_unparseable_last() -> None:
    completion = "Відповідь: 12\nВідповідь: <число>"
    result = extract_answer(completion)
    assert result.answer == "12"
    assert result.method is ExtractionMethod.MARKER


def test_markers_on_one_line_are_split() -> None:
    assert marker_segments("Відповідь: 5. Відповідь: 6", "Відповідь:") == ["5.", "6"]


def test_marker_inside_word_is_ignored() -> None:
    result = extract_answer("Невідповідь: 3", last_number_fallback=False)
    assert result.method is ExtractionMethod.FALLBACK


def test_custom_marker() -> None:
    assert extract_answer("Answer: 8", marker="Answer:").answer == "8"


def test_empty_marker_rejected() -> None:
    with pytest.raises(ValueError, match="keyword"):
        extract_answer("Відповідь: 1", marker=":")


def test_boxed_fallback_when_marker_missing() -> None:
    result = extract_answer(r"Отже, маємо $\boxed{\frac{18}{20}}$, тобто 0.9 частини.")
    assert result.answer == "9/10"
    assert result.method is ExtractionMethod.BOXED


def test_last_number_fallback_when_marker_missing() -> None:
    result = extract_answer("Спочатку 12 + 8 = 20, потім 20 × 3 = 60.")
    assert result.answer == "60"
    assert result.method is ExtractionMethod.LAST_NUMBER


def test_last_number_fallback_can_be_disabled() -> None:
    result = extract_answer("Спочатку 12 + 8 = 20.", last_number_fallback=False)
    assert result.answer is None
    assert result.method is ExtractionMethod.FALLBACK


def test_unparseable_marker_falls_back_to_last_number() -> None:
    result = extract_answer("Маємо 5 см і 3 мм.\nВідповідь: 5 см 3 мм")
    assert result.answer == "3"
    assert result.method is ExtractionMethod.LAST_NUMBER


@pytest.mark.parametrize("completion", ["", "   ", "Не знаю.", "Відповідь:"])
def test_missing_answer_returns_none(completion: str) -> None:
    result = extract_answer(completion)
    assert result.answer is None
    assert result.method is ExtractionMethod.FALLBACK


def test_last_boxed_content_handles_nesting() -> None:
    assert last_boxed_content(r"\boxed{1} і \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_last_boxed_content_unbalanced() -> None:
    assert last_boxed_content(r"\boxed{12") is None


def test_clean_latex_removes_wrappers() -> None:
    assert clean_latex(r"$\boxed{20{,}7}$").strip() == "20,7"
