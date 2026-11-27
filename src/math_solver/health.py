from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from math_solver.generation import FinishReason

EXPECTED_SCRIPTS = frozenset({"CYRILLIC", "LATIN", "GREEK"})
REPLACEMENT_CHARACTER = "�"
DIGIT_PLACEHOLDER = "#"
DIGIT_RUN_PATTERN = re.compile(r"\d+(?:[^\S\n]+\d+)*")
WHITESPACE_PATTERN = re.compile(r"\s+")


class HealthIssue(StrEnum):
    EMPTY = "empty"
    GARBLED = "garbled"
    REPETITIVE = "repetitive"
    TRUNCATED = "truncated"


GARBAGE_ISSUES = frozenset({HealthIssue.EMPTY, HealthIssue.GARBLED, HealthIssue.REPETITIVE})


@dataclass(frozen=True)
class HealthThresholds:
    min_letters_for_script_check: int = 20
    max_foreign_letter_ratio: float = 0.3
    max_replacement_characters: int = 2
    min_words_for_ngram_check: int = 40
    ngram_size: int = 4
    min_unique_ngram_ratio: float = 0.3
    tail_window_chars: int = 120
    max_tail_period: int = 30
    masked_tail_window_chars: int = 600
    min_masked_tail_repeats: int = 3


DEFAULT_THRESHOLDS = HealthThresholds()


@dataclass(frozen=True)
class CompletionCheck:
    text: str
    finish_reason: FinishReason
    invalid_tokens: int = 0


@dataclass(frozen=True)
class HealthSummary:
    problems: int
    completions: int
    issue_counts: dict[str, int]
    garbage_completions: int
    all_garbage_problems: int

    @property
    def garbage_ratio(self) -> float:
        return self.garbage_completions / self.completions if self.completions else 0.0

    def format(self) -> str:
        issues = ", ".join(f"{issue.value}={self.issue_counts.get(issue.value, 0)}" for issue in HealthIssue)
        return (
            f"health: problems={self.problems} completions={self.completions} {issues} "
            f"garbage={self.garbage_completions} ({100 * self.garbage_ratio:.1f}%) "
            f"all_garbage_problems={self.all_garbage_problems}"
        )


def assess_completion(
    check: CompletionCheck,
    thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
) -> frozenset[HealthIssue]:
    issues: set[HealthIssue] = set()
    if check.finish_reason is FinishReason.LENGTH:
        issues.add(HealthIssue.TRUNCATED)
    text = check.text.strip()
    if not text:
        issues.add(HealthIssue.EMPTY)
        return frozenset(issues)
    if check.invalid_tokens > 0 or _looks_garbled(text, thresholds):
        issues.add(HealthIssue.GARBLED)
    if _looks_repetitive(text, thresholds):
        issues.add(HealthIssue.REPETITIVE)
    return frozenset(issues)


def is_garbage(issues: Iterable[HealthIssue]) -> bool:
    return bool(GARBAGE_ISSUES.intersection(issues))


def summarize_health(
    problems: Iterable[Iterable[CompletionCheck]],
    thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
) -> HealthSummary:
    counts: Counter[str] = Counter()
    problem_count = completion_count = garbage = all_garbage = 0
    for checks in problems:
        assessed = [assess_completion(check, thresholds) for check in checks]
        problem_count += 1
        completion_count += len(assessed)
        flags = [is_garbage(issues) for issues in assessed]
        garbage += sum(flags)
        all_garbage += bool(flags) and all(flags)
        for issues in assessed:
            counts.update(issue.value for issue in issues)
    return HealthSummary(
        problems=problem_count,
        completions=completion_count,
        issue_counts=dict(counts),
        garbage_completions=garbage,
        all_garbage_problems=all_garbage,
    )


def _looks_garbled(text: str, thresholds: HealthThresholds) -> bool:
    if text.count(REPLACEMENT_CHARACTER) > thresholds.max_replacement_characters:
        return True
    letters = [character for character in text if character.isalpha()]
    if len(letters) < thresholds.min_letters_for_script_check:
        return False
    foreign = sum(_script(character) not in EXPECTED_SCRIPTS for character in letters)
    return foreign / len(letters) > thresholds.max_foreign_letter_ratio


def _script(character: str) -> str:
    name = unicodedata.name(character, "")
    return name.split(" ", 1)[0] if name else ""


def _looks_repetitive(text: str, thresholds: HealthThresholds) -> bool:
    return (
        _low_ngram_diversity(text, thresholds)
        or _has_periodic_tail(text, thresholds)
        or _has_masked_periodic_tail(text, thresholds)
    )


def _low_ngram_diversity(text: str, thresholds: HealthThresholds) -> bool:
    words = text.split()
    if len(words) < thresholds.min_words_for_ngram_check:
        return False
    size = thresholds.ngram_size
    ngrams = [tuple(words[index : index + size]) for index in range(len(words) - size + 1)]
    return len(set(ngrams)) / len(ngrams) < thresholds.min_unique_ngram_ratio


def _has_periodic_tail(text: str, thresholds: HealthThresholds) -> bool:
    window = thresholds.tail_window_chars
    if len(text) < window:
        return False
    tail = text[-window:]
    return any(
        all(tail[index] == tail[index + period] for index in range(window - period))
        for period in range(1, thresholds.max_tail_period + 1)
    )


def mask_digits(text: str) -> str:
    masked = DIGIT_RUN_PATTERN.sub(DIGIT_PLACEHOLDER, text)
    return WHITESPACE_PATTERN.sub(" ", masked).strip()


def _has_masked_periodic_tail(text: str, thresholds: HealthThresholds) -> bool:
    masked = mask_digits(text)
    window = thresholds.masked_tail_window_chars
    if len(masked) < window:
        return False
    tail = masked[-window:]
    max_period = window // thresholds.min_masked_tail_repeats
    return any(tail[period:] == tail[:-period] for period in range(1, max_period + 1))
