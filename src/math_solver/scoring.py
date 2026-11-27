from __future__ import annotations

import json
import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from math_solver.data_io import read_text_csv, require_columns
from math_solver.evaluate import EvaluationReport, evaluate_predictions, is_exact_match
from math_solver.extract import ExtractionMethod, ExtractionResult, extract_answer
from math_solver.config import ProjectConfig
from math_solver.health import HealthSummary
from math_solver.prompts import PromptTemplate
from math_solver.run_records import RunRecord, RunRecordError, read_records, require_consistent_signature
from math_solver.split import apply_split, load_split
from math_solver.tool_use import ExecutionStatus, SandboxSettings, ToolExecutionResult, solve_batch_with_tool
from math_solver.voting import VoteResult, log_agreement, vote_on_extractions

TOOL_METHOD = "tool"
METHOD_PRIORITY: tuple[str, ...] = (TOOL_METHOD, *(method.value for method in ExtractionMethod))

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoringOptions:
    marker: str
    use_tool: bool
    last_number_fallback: bool
    tool_text_fallback: bool
    sandbox: SandboxSettings


@dataclass(frozen=True)
class SampleAnswer:
    answer: str | None
    method: str
    tool_status: str | None = None
    tool_detail: str | None = None


@dataclass(frozen=True)
class ScoredProblem:
    id: str
    skill: str
    vote: VoteResult
    method: str
    samples: tuple[SampleAnswer, ...]
    completion: str

    @property
    def prediction(self) -> str | None:
        return self.vote.answer


@dataclass(frozen=True)
class Reference:
    answer: str
    statement: str


def load_run_file(path: Path) -> list[RunRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Run file not found: {path}")
    records_file = read_records(path)
    if records_file.truncated_tail_offset is not None:
        logger.warning("Ignoring a partially written last line in %s", path)
    validate_run(records_file.records)
    return records_file.records


def scoring_options(config: ProjectConfig, template: PromptTemplate, *, tool_text_fallback: bool) -> ScoringOptions:
    return ScoringOptions(
        marker=template.answer_marker,
        use_tool=template.tool_use is not None,
        last_number_fallback=config.extraction.last_number_fallback,
        tool_text_fallback=tool_text_fallback,
        sandbox=config.sandbox,
    )


def validate_run(records: Sequence[RunRecord]) -> None:
    if not records:
        raise RunRecordError("Run file contains no records")
    require_consistent_signature(records)
    expected_samples = records[0].generation_params.get("num_samples")
    wrong = [record.id for record in records if len(record.completions) != expected_samples]
    if wrong:
        raise RunRecordError(f"{len(wrong)} records do not have {expected_samples} completions, e.g. '{wrong[0]}'")


def score_records(records: Sequence[RunRecord], options: ScoringOptions) -> list[ScoredProblem]:
    flat = [completion for record in records for completion in record.completions]
    tool_results: list[ToolExecutionResult | None] = [None] * len(flat)
    if options.use_tool:
        tool_results = list(solve_batch_with_tool(flat, options.sandbox))
    scored: list[ScoredProblem] = []
    position = 0
    for record in records:
        count = len(record.completions)
        samples = [
            _sample_answer(completion, result, options)
            for completion, result in zip(record.completions, tool_results[position : position + count])
        ]
        position += count
        scored.append(_aggregate(record, samples))
    if any(problem.vote.num_samples > 1 for problem in scored):
        log_agreement([problem.vote for problem in scored])
    return scored


def tool_status_counts(scored: Sequence[ScoredProblem]) -> dict[str, int]:
    counts = Counter(sample.tool_status for problem in scored for sample in problem.samples if sample.tool_status)
    return dict(sorted(counts.items()))


def evaluate_scored(scored: Sequence[ScoredProblem], references: Mapping[str, Reference]) -> EvaluationReport:
    unknown = [problem.id for problem in scored if problem.id not in references]
    if unknown:
        raise RunRecordError(f"{len(unknown)} run ids are not in the validation split, e.g. '{unknown[0]}'")
    return evaluate_predictions(
        [problem.prediction for problem in scored],
        [references[problem.id].answer for problem in scored],
        [problem.skill for problem in scored],
        methods=[problem.method for problem in scored],
    )


def mispredicted_rows(
    scored: Sequence[ScoredProblem],
    references: Mapping[str, Reference],
) -> list[dict[str, Any]]:
    rows = []
    for problem in scored:
        reference = references[problem.id]
        if is_exact_match(problem.prediction, reference.answer):
            continue
        rows.append(
            {
                "id": problem.id,
                "skill": problem.skill,
                "statement": reference.statement,
                "reference": reference.answer,
                "prediction": problem.prediction,
                "method": problem.method,
                "votes": problem.vote.votes,
                "num_samples": problem.vote.num_samples,
                "distribution": [list(item) for item in problem.vote.distribution],
                "tool_statuses": [sample.tool_status for sample in problem.samples if sample.tool_status],
                "completion": problem.completion,
            }
        )
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def metrics_payload(
    report: EvaluationReport,
    *,
    run_name: str,
    val_rows: int,
    health: HealthSummary,
    tool_statuses: Mapping[str, int],
    scored: Sequence[ScoredProblem],
) -> dict[str, Any]:
    mean_agreement = sum(problem.vote.agreement for problem in scored) / len(scored)
    return {
        "run": run_name,
        "evaluated": report.total,
        "val_rows": val_rows,
        "accuracy": report.accuracy,
        "correct": report.correct,
        "unparsed_predictions": report.unparsed_predictions,
        "mean_agreement": mean_agreement,
        "per_skill": _frame_records(report.per_skill),
        "per_method": _frame_records(report.per_method),
        "health": {
            "completions": health.completions,
            "garbage_completions": health.garbage_completions,
            "all_garbage_problems": health.all_garbage_problems,
            "issues": health.issue_counts,
        },
        "tool_statuses": dict(tool_statuses),
    }


def _sample_answer(completion: str, tool_result: ToolExecutionResult | None, options: ScoringOptions) -> SampleAnswer:
    status = None if tool_result is None else tool_result.status.value
    detail = None if tool_result is None else tool_result.error or tool_result.raw_output
    if tool_result is not None and tool_result.status is ExecutionStatus.OK:
        return SampleAnswer(tool_result.answer, TOOL_METHOD, status, detail)
    if tool_result is not None and not options.tool_text_fallback:
        return SampleAnswer(None, ExtractionMethod.FALLBACK.value, status, detail)
    extraction = extract_answer(completion, marker=options.marker, last_number_fallback=options.last_number_fallback)
    return SampleAnswer(extraction.answer, extraction.method.value, status, detail)


def _aggregate(record: RunRecord, samples: Sequence[SampleAnswer]) -> ScoredProblem:
    vote = vote_on_extractions([ExtractionResult(sample.answer, _vote_method(sample.method)) for sample in samples])
    supporters = [
        index for index, sample in enumerate(samples) if vote.answer is not None and sample.answer == vote.answer
    ]
    method = (
        min((samples[index].method for index in supporters), key=METHOD_PRIORITY.index)
        if supporters
        else ExtractionMethod.FALLBACK.value
    )
    chosen = supporters[0] if supporters else 0
    return ScoredProblem(
        id=record.id,
        skill=record.skill,
        vote=vote,
        method=method,
        samples=tuple(samples),
        completion=record.completions[chosen],
    )


def _vote_method(method: str) -> ExtractionMethod:
    return ExtractionMethod.MARKER if method == TOOL_METHOD else ExtractionMethod(method)


def _frame_records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None:
        return []
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value


def load_val_references(train_csv: Path, split_file: Path) -> dict[str, Reference]:
    _, val = apply_split(read_text_csv(train_csv), load_split(split_file))
    require_columns(val, ["id", "answer", "statement"], "Validation split")
    return {
        str(row["id"]): Reference(answer=str(row["answer"]), statement=str(row["statement"]))
        for row in val.to_dict("records")
    }
