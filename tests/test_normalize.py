from __future__ import annotations

import pytest

from conftest import REPO_ROOT
from math_solver.data_io import read_text_csv
from math_solver.normalize import extract_canonical_numbers, normalize_answer

TRAIN_CSV = REPO_ROOT / "data" / "raw" / "train.csv"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1220", "1220"),
        ("  42  ", "42"),
        ("007", "7"),
        ("1 220", "1220"),
        ("1 220", "1220"),
        ("1 220", "1220"),
        ("1 000 000", "1000000"),
        ("17,7", "17.7"),
        ("17.7", "17.7"),
        ("3.50", "3.5"),
        ("3,50", "3.5"),
        ("3.0", "3"),
        ("0.500", "0.5"),
        ("1 220,50", "1220.5"),
        ("18/20", "9/10"),
        ("9/10", "9/10"),
        ("9 / 10", "9/10"),
        ("4/2", "2"),
        ("0/5", "0"),
        ("1 2/3", "5/3"),
        ("-12", "-12"),
        ("−12", "-12"),
        ("-3,50", "-3.5"),
        ("-3/6", "-1/2"),
        ("-0", "0"),
        ("-0.00", "0"),
        ("42 см", "42"),
        ("12см", "12"),
        ("12 м2", "12"),
        ("50%", "50"),
        ("Відповідь: 1 220 грн.", "1220"),
        ("Відповідь — 12", "12"),
        ("x = 7", "7"),
        (r"\frac{18}{20}", "9/10"),
        (r"\dfrac{3}{4}", "3/4"),
        (r"2\frac{1}{2}", "5/2"),
        (r"\boxed{15}", "15"),
    ],
)
def test_normalize_answer_canonical_forms(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "немає відповіді",
        "-",
        "3 і 5",
        "12 см, тобто 120 мм",
        "1,2,3",
        "5/0",
        "x2",
        "2 3",
    ],
)
def test_normalize_answer_rejects_malformed(raw: str | None) -> None:
    assert normalize_answer(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("988300", "988300"),
        ("988 300", "988300"),
        ("0,001012", "0.001012"),
        ("29,83", "29.83"),
        ("207/10", "207/10"),
        ("69/5", "69/5"),
        ("138/10", "69/5"),
        ("20 7/10", "207/10"),
        ("53/1000", "53/1000"),
    ],
)
def test_normalize_answer_train_answer_shapes(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


@pytest.mark.parametrize(("raw", "kept"), [("207/10", "207/10"), ("20.7", "20.7")])
def test_normalize_answer_never_converts_between_fraction_and_decimal(raw: str, kept: str) -> None:
    assert normalize_answer(raw) == kept


@pytest.mark.parametrize("canonical", ["0", "1220", "17.7", "9/10", "-1/2", "-3.5", "5/3"])
def test_normalize_answer_is_idempotent(canonical: str) -> None:
    assert normalize_answer(canonical) == canonical


def test_extract_canonical_numbers_returns_all_parseable_numbers() -> None:
    assert extract_canonical_numbers("12 + 8,50 = 20,5; дріб 18/20, ділення 5/0") == ["12", "8.5", "20.5", "9/10"]


@pytest.mark.skipif(not TRAIN_CSV.is_file(), reason="data/raw/train.csv is not available")
def test_all_train_gold_answers_are_already_canonical() -> None:
    answers = read_text_csv(TRAIN_CSV)["answer"].tolist()
    non_canonical = [answer for answer in answers if normalize_answer(answer) != answer]
    assert non_canonical == []
