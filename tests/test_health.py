from __future__ import annotations

from collections.abc import Callable

import pytest

from math_solver.generation import FinishReason
from math_solver.health import (
    CompletionCheck,
    HealthIssue,
    HealthThresholds,
    assess_completion,
    is_garbage,
    mask_digits,
    summarize_health,
)

STOP = FinishReason.STOP
LENGTH = FinishReason.LENGTH
MASKED_CHECK_DISABLED = HealthThresholds(masked_tail_window_chars=10**9)
HEALTHY = (
    "Щоб перевести години у хвилини, множимо на 60, бо в одній годині 60 хвилин.\n"
    "10 · 60 = 600.\n"
    "Відповідь: 600"
)


def _issues(
    text: str,
    finish_reason: FinishReason = STOP,
    invalid_tokens: int = 0,
    thresholds: HealthThresholds | None = None,
) -> frozenset[HealthIssue]:
    check = CompletionCheck(text, finish_reason, invalid_tokens)
    return assess_completion(check) if thresholds is None else assess_completion(check, thresholds)


def _counting_loop(last_step: int) -> str:
    lines = [
        "Розв'язуємо задачу крок за кроком:",
        "",
        "1.  Ділимо перші дві цифри числа (54) на 8:",
        "    54 ÷ 8 = 6 (остача 6)",
        "",
        "2.  Переносимо наступну цифру (4) до остачі:",
        "    6 (остача) 4",
        "",
    ]
    for step in range(3, last_step + 1, 2):
        fours = " ".join(["4"] * (step // 2))
        lines += [
            f"{step}. Ділимо першу цифру остачі (4) на 8:",
            "    4 ÷ 8 = 0 (остача 4)",
            "",
            f"{step + 1}. Переносимо наступну цифру (1) до остачі:",
            f"    0 (остача) 1 4 6 {fours}",
            "",
        ]
    lines.append(f"{last_step + 2}. Ділимо першу цифру остачі (4)")
    return "\n".join(lines)


def _long_division_solution() -> str:
    steps = [
        ("9", "3", "3", "9", "0"),
        ("8", "3", "2", "6", "2"),
        ("27", "3", "9", "27", "0"),
        ("6", "3", "2", "6", "0"),
        ("5", "3", "1", "3", "2"),
        ("24", "3", "8", "24", "0"),
    ]
    lines = ["Ділимо 987654 на 3 куточком, записуючи кожен крок."]
    for index, (part, divisor, digit, product, remainder) in enumerate(steps, start=1):
        lines += [
            f"{index}. Беремо неповне ділене {part}.",
            f"   {part} : {divisor} = {digit}, бо {digit} · {divisor} = {product}.",
            f"   Остача: {part} - {product} = {remainder}. Зносимо наступну цифру.",
        ]
    lines += [
        "Записуємо цифри частки по порядку: 3, 2, 9, 2, 1, 8.",
        "Перевірка: 329218 · 3 = 987654, тож ділення виконано правильно.",
        "Відповідь: 329218",
    ]
    return "\n".join(lines)


def _divisibility_enumeration() -> str:
    numbers = ["250", "317", "243", "612", "948", "352", "420", "729"]
    lines = ["Число ділиться на 9, якщо сума його цифр ділиться на 9. Перевіримо кожне число."]
    for index, number in enumerate(numbers, start=1):
        digit_sum = sum(int(digit) for digit in number)
        verdict = "ділиться" if digit_sum % 9 == 0 else "не ділиться"
        lines += [
            f"{index}. Число {number}:",
            f"   Сума цифр: {' + '.join(number)} = {digit_sum}",
            f"   {digit_sum} {verdict} на 9, тому {number} {verdict} на 9.",
        ]
    lines += ["На 9 діляться числа 243, 612 і 729, тобто три числа.", "Відповідь: 3"]
    return "\n".join(lines)


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


def test_mask_digits_collapses_numbers_and_whitespace() -> None:
    assert mask_digits("17. Крок:\n   0 (остача) 1 4 6 4 4 4") == "#. Крок: # (остача) #"
    assert mask_digits("1,012 : 1000 = 0.001012") == "#,# : # = #.#"


def test_counting_loop_with_growing_digit_runs_is_repetitive() -> None:
    text = _counting_loop(last_step=26)
    issues = _issues(text, LENGTH)
    assert issues == {HealthIssue.REPETITIVE, HealthIssue.TRUNCATED}
    assert is_garbage(issues)


def test_counting_loop_is_missed_without_digit_masking() -> None:
    assert _issues(_counting_loop(last_step=26), LENGTH, thresholds=MASKED_CHECK_DISABLED) == {
        HealthIssue.TRUNCATED
    }


def test_inline_counting_loop_is_repetitive() -> None:
    text = "Рахуємо далі: " + " ".join(f"крок {step}: додаємо 1 і маємо {step + 1}." for step in range(1, 60))
    assert HealthIssue.REPETITIVE in _issues(text, LENGTH)


def test_short_counting_pattern_is_below_masked_length_guard() -> None:
    text = _counting_loop(last_step=6)
    assert len(mask_digits(text)) < HealthThresholds().masked_tail_window_chars
    assert _issues(text, LENGTH) == {HealthIssue.TRUNCATED}


@pytest.mark.parametrize("build", [_long_division_solution, _divisibility_enumeration])
def test_legitimate_multi_step_arithmetic_is_not_flagged(build: Callable[[], str]) -> None:
    text = build()
    assert len(mask_digits(text)) >= HealthThresholds().masked_tail_window_chars
    assert _issues(text) == frozenset()


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
