from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from math_solver.normalize import extract_canonical_numbers, normalize_answer

DEFAULT_ANSWER_MARKER = "Відповідь:"

_MARKER_TERMINATORS = ":：—–"
_MARKER_SUFFIX = rf"\s*\**\s*[{_MARKER_TERMINATORS}]\**"
_UNWRAP_PATTERN = re.compile(r"\\(?:boxed|fbox|text|textbf|textit|textrm|mathrm|mathbf)\s*\{")
_BOXED_PATTERN = re.compile(r"\\(?:boxed|fbox)\s*\{")
_LATEX_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("{,}", ","),
    ("{.}", "."),
    ("\\left", ""),
    ("\\right", ""),
    ("\\,", " "),
    ("\\;", " "),
    ("\\:", " "),
    ("\\!", ""),
    ("\\ ", " "),
    ("\\%", "%"),
    ("\\times", " × "),
    ("\\cdot", " · "),
    ("\\div", " : "),
    ("\\(", " "),
    ("\\)", " "),
    ("\\[", " "),
    ("\\]", " "),
    ("$", " "),
    ("**", " "),
    ("`", " "),
)


class ExtractionMethod(StrEnum):
    MARKER = "marker"
    BOXED = "boxed"
    LAST_NUMBER = "last_number"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class ExtractionResult:
    answer: str | None
    method: ExtractionMethod


def extract_answer(
    completion: str,
    *,
    marker: str = DEFAULT_ANSWER_MARKER,
    last_number_fallback: bool = True,
) -> ExtractionResult:
    for segment in reversed(marker_segments(completion, marker)):
        answer = parse_answer_segment(segment)
        if answer is not None:
            return ExtractionResult(answer, ExtractionMethod.MARKER)
    boxed = last_boxed_content(completion)
    if boxed is not None:
        answer = parse_answer_segment(boxed)
        if answer is not None:
            return ExtractionResult(answer, ExtractionMethod.BOXED)
    if last_number_fallback:
        numbers = extract_canonical_numbers(clean_latex(completion))
        if numbers:
            return ExtractionResult(numbers[-1], ExtractionMethod.LAST_NUMBER)
    return ExtractionResult(None, ExtractionMethod.FALLBACK)


def marker_segments(completion: str, marker: str) -> list[str]:
    matches = list(_marker_pattern(marker).finditer(completion))
    segments = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(completion)
        segments.append(_first_non_empty_line(completion[match.end() : end]))
    return segments


def parse_answer_segment(segment: str) -> str | None:
    text = clean_latex(segment)
    if "=" in text:
        text = text.rsplit("=", 1)[1]
    return normalize_answer(text)


def clean_latex(text: str) -> str:
    cleaned = _unwrap_commands(text, _UNWRAP_PATTERN)
    for old, new in _LATEX_REPLACEMENTS:
        cleaned = cleaned.replace(old, new)
    return cleaned


def last_boxed_content(text: str) -> str | None:
    matches = list(_BOXED_PATTERN.finditer(text))
    if not matches:
        return None
    opening = matches[-1].end() - 1
    closing = _matching_brace(text, opening)
    if closing is None:
        return None
    return text[opening + 1 : closing]


def _marker_pattern(marker: str) -> re.Pattern[str]:
    keyword = marker.strip().rstrip(_MARKER_TERMINATORS).strip()
    if not keyword:
        raise ValueError(f"Answer marker must contain a keyword, got '{marker}'")
    return re.compile(rf"(?<!\w)\**{re.escape(keyword)}{_MARKER_SUFFIX}", re.IGNORECASE)


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _unwrap_commands(text: str, pattern: re.Pattern[str]) -> str:
    while (match := pattern.search(text)) is not None:
        opening = match.end() - 1
        closing = _matching_brace(text, opening)
        if closing is None:
            text = text[: match.start()] + " " + text[match.end() :]
            continue
        text = text[: match.start()] + " " + text[opening + 1 : closing] + " " + text[closing + 1 :]
    return text


def _matching_brace(text: str, opening: int) -> int | None:
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return position
    return None
