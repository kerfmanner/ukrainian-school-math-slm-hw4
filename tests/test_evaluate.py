from __future__ import annotations

import math

import pytest

from math_solver.evaluate import evaluate_completions, evaluate_predictions, is_exact_match


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("17,7", "17.7", True),
        ("3.50", "3.5", True),
        ("18/20", "9/10", True),
        ("1 220 грн", "1220", True),
        ("0.9", "9/10", False),
        ("12", "13", False),
        (None, "12", False),
        ("немає", "12", False),
        ("Так", "так", True),
        (None, "так", False),
        ("", "", False),
    ],
)
def test_is_exact_match(prediction: str | None, reference: str, expected: bool) -> None:
    assert is_exact_match(prediction, reference) is expected


def test_evaluate_predictions_overall_and_per_skill() -> None:
    report = evaluate_predictions(
        predictions=["1", "2,5", None, "4/8", "x"],
        references=["1", "2.5", "3", "1/2", "7"],
        skills=["b", "b", "a", "a", "c"],
    )
    assert report.total == 5
    assert report.correct == 3
    assert report.accuracy == pytest.approx(0.6)
    assert report.unparsed_predictions == 2
    assert report.unparsed_references == 0
    per_skill = report.per_skill.set_index("skill")
    assert per_skill.index.tolist() == ["a", "b", "c"]
    assert per_skill.loc["a", "total"] == 2
    assert per_skill.loc["a", "correct"] == 1
    assert per_skill.loc["a", "accuracy"] == pytest.approx(0.5)
    assert per_skill.loc["b", "accuracy"] == pytest.approx(1.0)
    assert per_skill.loc["c", "accuracy"] == pytest.approx(0.0)


def test_evaluate_predictions_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_predictions(["1"], ["1", "2"], ["a", "a"])


def test_evaluate_predictions_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate_predictions([], [], [])


def test_evaluate_predictions_breaks_down_by_method() -> None:
    report = evaluate_predictions(
        predictions=["1", "2", "9", None],
        references=["1", "2", "3", "4"],
        skills=["a", "a", "b", "b"],
        methods=["marker", "marker", "last_number", "fallback"],
    )
    per_method = report.per_method
    assert per_method is not None
    assert per_method["method"].tolist() == ["marker", "boxed", "last_number", "fallback"]
    rows = per_method.set_index("method")
    assert rows.loc["marker", "total"] == 2
    assert rows.loc["marker", "accuracy"] == 1.0
    assert rows.loc["last_number", "correct"] == 0
    assert rows.loc["fallback", "total"] == 1
    assert rows.loc["boxed", "total"] == 0
    assert math.isnan(rows.loc["boxed", "accuracy"])


def test_evaluate_predictions_without_methods_has_no_breakdown() -> None:
    assert evaluate_predictions(["1"], ["1"], ["a"]).per_method is None


def test_evaluate_predictions_rejects_method_length_mismatch() -> None:
    with pytest.raises(ValueError, match="methods"):
        evaluate_predictions(["1"], ["1"], ["a"], methods=["marker", "marker"])


def test_evaluate_completions_extracts_and_tracks_methods() -> None:
    report = evaluate_completions(
        completions=["Розв'язання...\nВідповідь: 20,7", "Отже 5 + 7 = 12", r"Маємо $\boxed{9/10}$", "Не знаю"],
        references=["20.7", "12", "9/10", "4"],
        skills=["D", "N", "F", "N"],
    )
    assert report.correct == 3
    rows = report.per_method.set_index("method")
    assert rows["total"].to_dict() == {"marker": 1, "boxed": 1, "last_number": 1, "fallback": 1}
    assert rows.loc["fallback", "accuracy"] == 0.0
