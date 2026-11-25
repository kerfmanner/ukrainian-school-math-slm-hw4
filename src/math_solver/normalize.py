from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction

_GROUP_SEPARATORS = r"\u0020\u00a0\u202f\u2009"
_GROUP_SEPARATOR_PATTERN = re.compile(rf"[{_GROUP_SEPARATORS}]")
_LATEX_FRACTION_PATTERN = re.compile(r"\\[dt]?frac\s*\{\s*([^{}]*?)\s*\}\s*\{\s*([^{}]*?)\s*\}")
_NUMBER_PATTERN = re.compile(
    rf"""
    (?<!\w)
    (?P<sign>[-\u2212])?
    (?:
        (?P<whole>[0-9]+)[{_GROUP_SEPARATORS}]+
        (?P<mixed_numerator>[0-9]+)[{_GROUP_SEPARATORS}]*/[{_GROUP_SEPARATORS}]*(?P<mixed_denominator>[0-9]+)
      | (?P<numerator>[0-9]+)[{_GROUP_SEPARATORS}]*/[{_GROUP_SEPARATORS}]*(?P<denominator>[0-9]+)
      | (?P<grouped>[0-9]{{1,3}}(?:[{_GROUP_SEPARATORS}][0-9]{{3}})+(?![0-9])(?:[.,][0-9]+)?)
      | (?P<plain>[0-9]+(?:[.,][0-9]+)?)
    )
    """,
    re.VERBOSE,
)


def normalize_answer(raw: str | None) -> str | None:
    if raw is None:
        return None
    candidates = _canonical_candidates(raw)
    if len(candidates) != 1:
        return None
    return candidates[0]


def extract_canonical_numbers(text: str) -> list[str]:
    return [candidate for candidate in _canonical_candidates(text) if candidate is not None]


def _canonical_candidates(text: str) -> list[str | None]:
    prepared = _LATEX_FRACTION_PATTERN.sub(r" \1/\2", text)
    return [_canonicalize(match) for match in _NUMBER_PATTERN.finditer(prepared)]


def _canonicalize(match: re.Match[str]) -> str | None:
    value = _parse_value(match)
    if value is None:
        return None
    if match["sign"] is not None:
        value = -value
    if isinstance(value, Fraction):
        return _format_fraction(value)
    return _format_decimal(value)


def _parse_value(match: re.Match[str]) -> Fraction | Decimal | None:
    if match["denominator"] is not None:
        return _safe_fraction(int(match["numerator"]), int(match["denominator"]))
    if match["mixed_denominator"] is not None:
        fractional_part = _safe_fraction(int(match["mixed_numerator"]), int(match["mixed_denominator"]))
        if fractional_part is None:
            return None
        return int(match["whole"]) + fractional_part
    digits = match["grouped"] or match["plain"]
    return Decimal(_GROUP_SEPARATOR_PATTERN.sub("", digits).replace(",", "."))


def _safe_fraction(numerator: int, denominator: int) -> Fraction | None:
    if denominator == 0:
        return None
    return Fraction(numerator, denominator)


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text == "-0" else text
