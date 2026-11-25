from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from math_solver.extract import DEFAULT_ANSWER_MARKER, ExtractionMethod, extract_answer
from math_solver.normalize import normalize_answer

BREAKDOWN_COLUMNS: tuple[str, ...] = ("total", "correct", "accuracy")


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    total: int
    correct: int
    unparsed_predictions: int
    unparsed_references: int
    per_skill: pd.DataFrame
    per_method: pd.DataFrame | None = None


def is_exact_match(prediction: str | None, reference: str) -> bool:
    expected = normalize_answer(reference)
    if expected is not None:
        return normalize_answer(prediction) == expected
    if prediction is None or not reference.strip():
        return False
    return _collapse_text(prediction) == _collapse_text(reference)


def evaluate_predictions(
    predictions: Sequence[str | None],
    references: Sequence[str],
    skills: Sequence[str],
    methods: Sequence[str] | None = None,
) -> EvaluationReport:
    if not len(predictions) == len(references) == len(skills):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, references={len(references)}, skills={len(skills)}"
        )
    if methods is not None and len(methods) != len(predictions):
        raise ValueError(f"Length mismatch: methods={len(methods)}, predictions={len(predictions)}")
    if not references:
        raise ValueError("Cannot evaluate an empty set of predictions")
    frame = pd.DataFrame(
        {
            "skill": list(skills),
            "correct": [is_exact_match(prediction, reference) for prediction, reference in zip(predictions, references)],
        }
    )
    per_method = None
    if methods is not None:
        frame["method"] = [str(method) for method in methods]
        per_method = _method_breakdown(frame)
    correct = int(frame["correct"].sum())
    return EvaluationReport(
        accuracy=correct / len(frame),
        total=len(frame),
        correct=correct,
        unparsed_predictions=sum(normalize_answer(prediction) is None for prediction in predictions),
        unparsed_references=sum(normalize_answer(reference) is None for reference in references),
        per_skill=_breakdown(frame, "skill"),
        per_method=per_method,
    )


def evaluate_completions(
    completions: Sequence[str],
    references: Sequence[str],
    skills: Sequence[str],
    *,
    marker: str = DEFAULT_ANSWER_MARKER,
    last_number_fallback: bool = True,
) -> EvaluationReport:
    extractions = [
        extract_answer(completion, marker=marker, last_number_fallback=last_number_fallback)
        for completion in completions
    ]
    return evaluate_predictions(
        [extraction.answer for extraction in extractions],
        references,
        skills,
        methods=[extraction.method.value for extraction in extractions],
    )


def _breakdown(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = frame.groupby(column, sort=True)["correct"].agg(total="size", correct="sum").reset_index()
    grouped["accuracy"] = grouped["correct"] / grouped["total"]
    return grouped


def _method_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    observed = _breakdown(frame, "method").set_index("method")
    known = [method.value for method in ExtractionMethod]
    extra = sorted(set(observed.index) - set(known))
    ordered = observed.reindex(known + extra)
    ordered["total"] = ordered["total"].fillna(0).astype(int)
    ordered["correct"] = ordered["correct"].fillna(0).astype(int)
    ordered["accuracy"] = ordered["correct"] / ordered["total"].where(ordered["total"] > 0)
    return ordered.reset_index()[["method", *BREAKDOWN_COLUMNS]]


def _collapse_text(text: str) -> str:
    return " ".join(text.split()).casefold()
