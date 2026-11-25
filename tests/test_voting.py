from __future__ import annotations

import logging

import pytest

from math_solver.extract import ExtractionMethod, ExtractionResult
from math_solver.voting import aggregate_self_consistency, majority_vote, vote_on_extractions

MARKER = ExtractionMethod.MARKER
LAST = ExtractionMethod.LAST_NUMBER
BOXED = ExtractionMethod.BOXED


def _votes(*items: tuple[str | None, ExtractionMethod]) -> list[ExtractionResult]:
    return [ExtractionResult(answer, method) for answer, method in items]


def test_majority_vote_normalizes_before_counting() -> None:
    completions = ["Відповідь: 20,7", "Відповідь: 20.70", "Відповідь: 21", "Відповідь: $20{,}7$", "Відповідь: 21"]
    result = majority_vote(completions)
    assert result.answer == "20.7"
    assert result.votes == 3
    assert result.num_samples == 5
    assert result.agreement == pytest.approx(0.6)
    assert result.distribution == (("20.7", 3), ("21", 2))
    assert result.method is MARKER


def test_unanswered_samples_count_against_agreement() -> None:
    result = majority_vote(["Відповідь: 5", "не знаю", "Відповідь: 5", "???"], last_number_fallback=False)
    assert result.answer == "5"
    assert result.agreement == pytest.approx(0.5)


def test_tie_prefers_marker_over_last_number() -> None:
    result = vote_on_extractions(_votes(("7", LAST), ("7", LAST), ("8", MARKER), ("8", LAST)))
    assert result.answer == "8"
    assert result.method is MARKER


def test_tie_prefers_boxed_over_last_number() -> None:
    result = vote_on_extractions(_votes(("1", LAST), ("2", BOXED)))
    assert result.answer == "2"
    assert result.method is BOXED


def test_majority_beats_method_preference() -> None:
    result = vote_on_extractions(_votes(("3", LAST), ("3", LAST), ("4", MARKER)))
    assert result.answer == "3"
    assert result.method is LAST


def test_full_tie_breaks_by_first_occurrence() -> None:
    result = vote_on_extractions(_votes(("9", MARKER), ("1", MARKER)))
    assert result.answer == "9"
    reversed_result = vote_on_extractions(_votes(("1", MARKER), ("9", MARKER)))
    assert reversed_result.answer == "1"


def test_winner_method_is_best_among_its_votes() -> None:
    result = vote_on_extractions(_votes(("5", LAST), ("5", MARKER), ("6", LAST)))
    assert result.method is MARKER


def test_no_answers_falls_back() -> None:
    result = vote_on_extractions(_votes((None, ExtractionMethod.FALLBACK), (None, ExtractionMethod.FALLBACK)))
    assert result.answer is None
    assert result.method is ExtractionMethod.FALLBACK
    assert result.votes == 0
    assert result.agreement == 0.0
    assert result.distribution == ()


def test_empty_sample_list() -> None:
    result = vote_on_extractions([])
    assert result.answer is None
    assert result.agreement == 0.0


def test_aggregate_logs_agreement(caplog: pytest.LogCaptureFixture) -> None:
    groups = [
        ["Відповідь: 1", "Відповідь: 1"],
        ["Відповідь: 2", "Відповідь: 3"],
        ["нічого", "теж нічого"],
    ]
    with caplog.at_level(logging.INFO):
        results = aggregate_self_consistency(groups, expected_samples=2)
    assert [result.answer for result in results] == ["1", "2", None]
    assert "mean agreement 0.500, unanimous 1, no answer 1" in caplog.text


def test_aggregate_checks_sample_counts() -> None:
    with pytest.raises(ValueError, match="do not have 3 samples"):
        aggregate_self_consistency([["Відповідь: 1"] * 3, ["Відповідь: 1"] * 2], expected_samples=3)
