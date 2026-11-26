from __future__ import annotations

import pytest

from math_solver.generation import FinishReason
from math_solver.health import CompletionCheck, HealthIssue, assess_completion, is_garbage, summarize_health

STOP = FinishReason.STOP
HEALTHY = (
    "Щоб перевести години у хвилини, множимо на 60, бо в одній годині 60 хвилин.\n"
    "10 · 60 = 600.\n"
    "Відповідь: 600"
)


def _issues(text: str, finish_reason: FinishReason = STOP, invalid_tokens: int = 0) -> frozenset[HealthIssue]:
    return assess_completion(CompletionCheck(text, finish_reason, invalid_tokens))


def test_healthy_ukrainian_solution_has_no_issues() -> None:
    assert _issues(HEALTHY) == frozenset()


def test_latex_and_python_are_not_garbled() -> None:
    text = "Маємо $\\frac{3}{4} \\cdot 12 = 9$.\n```python\nanswer = 3 * 12 // 4\n```\nВідповідь: 9"
    assert _issues(text) == frozenset()


@pytest.mark.parametrize("text", ["", "   \n\t"])
def test_empty_completion(text: str) -> None:
    assert _issues(text) == {HealthIssue.EMPTY}


@pytest.mark.parametrize(
    "text",
    [
        "ประเทศไทย 漢字漢字漢字 ประเทศไทย ქართული ენა ประเทศไทย 漢字漢字 ქართული",
        "Відповідь ���� 12",
    ],
)
def test_garbled_text_is_flagged(text: str) -> None:
    assert HealthIssue.GARBLED in _issues(text)


def test_pad_tokens_inside_completion_flag_nan_garbage() -> None:
    assert HealthIssue.GARBLED in _issues("Відповідь: 5", invalid_tokens=3)


def test_word_loop_is_repetitive() -> None:
    text = "Отже, 5 + 5 = 10. " * 30
    assert HealthIssue.REPETITIVE in _issues(text)


def test_character_loop_without_spaces_is_repetitive() -> None:
    assert HealthIssue.REPETITIVE in _issues("Відповідь: " + "12" * 100)


def test_truncation_alone_is_not_garbage() -> None:
    issues = _issues(HEALTHY, FinishReason.LENGTH)
    assert issues == {HealthIssue.TRUNCATED}
    assert not is_garbage(issues)


def test_summary_counts_issues_and_all_garbage_problems() -> None:
    summary = summarize_health(
        [
            [CompletionCheck(HEALTHY, STOP), CompletionCheck("", FinishReason.LENGTH)],
            [CompletionCheck("", STOP), CompletionCheck("", STOP)],
            [CompletionCheck(HEALTHY, STOP)],
        ]
    )
    assert summary.problems == 3
    assert summary.completions == 5
    assert summary.garbage_completions == 3
    assert summary.all_garbage_problems == 1
    assert summary.issue_counts == {"empty": 3, "truncated": 1}
    assert "garbage=3 (60.0%)" in summary.format()
