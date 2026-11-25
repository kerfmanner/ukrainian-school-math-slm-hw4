from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from math_solver.data_io import read_text_csv, require_columns
from math_solver.normalize import normalize_answer

ID_COLUMN = "id"
ANSWER_COLUMN = "answer"
SUBMISSION_COLUMNS: tuple[str, ...] = (ID_COLUMN, ANSWER_COLUMN)
MAX_REPORTED_EXAMPLES = 3

logger = logging.getLogger(__name__)


class SubmissionError(ValueError):
    pass


@dataclass(frozen=True)
class SubmissionCheck:
    rows: int
    expected_rows: int
    missing_count: int
    non_canonical_count: int
    fallback_count: int
    problems: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.problems


def build_submission(
    sample: pd.DataFrame,
    predictions: Mapping[str, str | None],
    fallback_answer: str,
) -> pd.DataFrame:
    if normalize_answer(fallback_answer) != fallback_answer:
        raise SubmissionError(f"Fallback answer '{fallback_answer}' is not in canonical form")
    sample_ids = _sample_ids(sample)
    unknown_ids = sorted(set(predictions) - set(sample_ids))
    if unknown_ids:
        raise SubmissionError(f"{len(unknown_ids)} prediction ids are not in the sample submission: {unknown_ids[:5]}")
    normalized = [normalize_answer(predictions.get(row_id)) for row_id in sample_ids]
    fallback_count = sum(answer is None for answer in normalized)
    logger.info("Fallback answer used for %d/%d rows", fallback_count, len(sample_ids))
    submission = pd.DataFrame(
        {
            ID_COLUMN: sample_ids,
            ANSWER_COLUMN: [fallback_answer if answer is None else answer for answer in normalized],
        }
    )
    validate_submission(submission, sample)
    return submission


def check_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    fallback_answer: str | None = None,
) -> SubmissionCheck:
    sample_ids = _sample_ids(sample)
    problems: list[str] = []
    if expected_rows is not None and len(sample_ids) != expected_rows:
        problems.append(f"Sample submission has {len(sample_ids)} rows, expected {expected_rows}")
    if tuple(submission.columns) != SUBMISSION_COLUMNS:
        problems.append(f"Submission columns must be {list(SUBMISSION_COLUMNS)}, got {list(submission.columns)}")
        return SubmissionCheck(len(submission), len(sample_ids), 0, 0, 0, tuple(problems))
    if len(submission) != len(sample_ids):
        problems.append(f"Submission has {len(submission)} rows, expected {len(sample_ids)}")
    id_problem = _first_id_mismatch(submission[ID_COLUMN].astype(str).tolist(), sample_ids)
    if id_problem is not None:
        problems.append(id_problem)
    answers = submission[ANSWER_COLUMN]
    missing = answers.isna() | (answers.astype(str).str.strip() == "")
    if missing.any():
        problems.append(f"Submission has {int(missing.sum())} missing answers")
    present = [str(answer) for answer in answers[~missing]]
    non_canonical = [answer for answer in present if normalize_answer(answer) != answer]
    if non_canonical:
        problems.append(
            f"Submission has {len(non_canonical)} answers not in canonical form, "
            f"e.g. {non_canonical[:MAX_REPORTED_EXAMPLES]}"
        )
    fallback_count = 0 if fallback_answer is None else sum(answer == fallback_answer for answer in present)
    return SubmissionCheck(
        rows=len(submission),
        expected_rows=len(sample_ids),
        missing_count=int(missing.sum()),
        non_canonical_count=len(non_canonical),
        fallback_count=fallback_count,
        problems=tuple(problems),
    )


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    expected_rows: int | None = None,
) -> None:
    check = check_submission(submission, sample, expected_rows=expected_rows)
    if not check.is_valid:
        raise SubmissionError("; ".join(check.problems))


def save_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    path: Path,
    expected_rows: int | None = None,
) -> None:
    validate_submission(submission, sample, expected_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    validate_submission(read_text_csv(path), sample, expected_rows)
    logger.info("Saved submission with %d rows to %s", len(submission), path)


def _sample_ids(sample: pd.DataFrame) -> list[str]:
    require_columns(sample, [ID_COLUMN], "Sample submission")
    ids = sample[ID_COLUMN].astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise SubmissionError("Sample submission contains duplicate ids")
    return ids


def _first_id_mismatch(submission_ids: list[str], sample_ids: list[str]) -> str | None:
    for position, (actual, expected) in enumerate(zip(submission_ids, sample_ids)):
        if actual != expected:
            return f"Id mismatch at row {position}: got '{actual}', expected '{expected}'"
    return None
