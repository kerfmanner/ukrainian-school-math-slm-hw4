from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from math_solver.data_io import read_text_csv
from math_solver.submission import (
    SubmissionError,
    build_submission,
    check_submission,
    save_submission,
    validate_submission,
)


def test_build_submission_follows_sample_order_and_normalizes(
    sample_submission: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    predictions = {"3": "3,50", "101": "18/20", "55": "немає відповіді"}
    with caplog.at_level(logging.INFO):
        submission = build_submission(sample_submission, predictions, fallback_answer="0")
    assert submission.columns.tolist() == ["id", "answer"]
    assert submission["id"].tolist() == ["101", "7", "55", "3"]
    assert submission["answer"].tolist() == ["9/10", "0", "0", "3.5"]
    assert "Fallback answer used for 2/4 rows" in caplog.text


def test_build_submission_uses_configured_fallback(sample_submission: pd.DataFrame) -> None:
    submission = build_submission(sample_submission, {}, fallback_answer="1")
    assert submission["answer"].tolist() == ["1", "1", "1", "1"]


def test_build_submission_rejects_unknown_ids(sample_submission: pd.DataFrame) -> None:
    with pytest.raises(SubmissionError, match="not in the sample submission"):
        build_submission(sample_submission, {"999": "1"}, fallback_answer="0")


def test_build_submission_rejects_non_canonical_fallback(sample_submission: pd.DataFrame) -> None:
    with pytest.raises(SubmissionError, match="canonical"):
        build_submission(sample_submission, {}, fallback_answer="0,0")


def test_validate_submission_rejects_wrong_columns(sample_submission: pd.DataFrame) -> None:
    submission = sample_submission.rename(columns={"answer": "prediction"})
    with pytest.raises(SubmissionError, match="columns"):
        validate_submission(submission, sample_submission)


def test_validate_submission_rejects_extra_index_column(sample_submission: pd.DataFrame) -> None:
    submission = sample_submission.reset_index()
    with pytest.raises(SubmissionError, match="columns"):
        validate_submission(submission, sample_submission)


def test_validate_submission_rejects_wrong_order(sample_submission: pd.DataFrame) -> None:
    submission = sample_submission.iloc[::-1].reset_index(drop=True)
    with pytest.raises(SubmissionError, match="Id mismatch at row 0"):
        validate_submission(submission, sample_submission)


def test_validate_submission_rejects_row_count_mismatch(sample_submission: pd.DataFrame) -> None:
    with pytest.raises(SubmissionError, match="rows"):
        validate_submission(sample_submission.iloc[:3], sample_submission)


def test_validate_submission_rejects_unexpected_sample_size(sample_submission: pd.DataFrame) -> None:
    with pytest.raises(SubmissionError, match="expected 3561"):
        validate_submission(sample_submission, sample_submission, expected_rows=3561)


@pytest.mark.parametrize("bad_answer", [np.nan, None, "", "  "])
def test_validate_submission_rejects_missing_answers(sample_submission: pd.DataFrame, bad_answer: object) -> None:
    submission = sample_submission.astype(object)
    submission.loc[1, "answer"] = bad_answer
    with pytest.raises(SubmissionError, match="missing answers"):
        validate_submission(submission, sample_submission)


def test_validate_submission_rejects_duplicate_sample_ids() -> None:
    sample = pd.DataFrame({"id": ["1", "1"], "answer": ["0", "0"]})
    with pytest.raises(SubmissionError, match="duplicate"):
        validate_submission(sample, sample)


def test_save_submission_writes_exact_schema_without_index(sample_submission: pd.DataFrame, tmp_path: Path) -> None:
    submission = build_submission(sample_submission, {"7": "12"}, fallback_answer="0")
    output = tmp_path / "nested" / "submission.csv"
    save_submission(submission, sample_submission, output, expected_rows=4)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines == ["id,answer", "101,0", "7,12", "55,0", "3,0"]
    pd.testing.assert_frame_equal(read_text_csv(output), submission)


def test_check_submission_counts_fallbacks_and_collects_all_problems(sample_submission: pd.DataFrame) -> None:
    submission = pd.DataFrame({"id": ["101", "55", "7", "3"], "answer": ["0", "17,7", "", "0"]})
    check = check_submission(submission, sample_submission, expected_rows=4, fallback_answer="0")
    assert not check.is_valid
    assert check.fallback_count == 2
    assert check.missing_count == 1
    assert check.non_canonical_count == 1
    assert len(check.problems) == 3
    assert check.problems[0].startswith("Id mismatch at row 1")


def test_check_submission_accepts_valid_frame(sample_submission: pd.DataFrame) -> None:
    submission = build_submission(sample_submission, {"7": "12", "3": "9/10"}, fallback_answer="0")
    check = check_submission(submission, sample_submission, expected_rows=4, fallback_answer="0")
    assert check.is_valid
    assert check.fallback_count == 2
    assert check.rows == check.expected_rows == 4


def test_validate_submission_rejects_non_canonical_answers(sample_submission: pd.DataFrame) -> None:
    submission = sample_submission.copy()
    submission.loc[0, "answer"] = "3.50"
    with pytest.raises(SubmissionError, match="canonical"):
        validate_submission(submission, sample_submission)
