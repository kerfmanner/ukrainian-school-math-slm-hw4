from __future__ import annotations

import html
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from math_solver.config import ProjectConfig
from math_solver.evaluate import is_exact_match
from math_solver.health import assess_completion, is_garbage, summarize_health
from math_solver.inference import VAL_SPLIT, Problem, load_problems
from math_solver.normalize import normalize_answer
from math_solver.prompts import load_prompt
from math_solver.run_records import RunRecord
from math_solver.scoring import (
    Reference,
    ScoredProblem,
    evaluate_scored,
    load_run_file,
    load_val_references,
    metrics_payload,
    score_records,
    scoring_options,
    tool_status_counts,
)
from math_solver.tool_use import extract_python_code

TEMPLATE_PATH = Path(__file__).with_name("run_report_template.html")
DATA_PLACEHOLDER = "__REPORT_DATA__"
TITLE_PLACEHOLDER = "__REPORT_TITLE__"
NAME_SEPARATOR = "__"
COMPARISON_SEPARATOR = "__vs__"
REPORT_SUFFIX = "__report.html"
TOKEN_HISTOGRAM_BINS = 16
OMITTED_NOTICE = "\n…[{omitted} characters omitted]"
JSON_HTML_ESCAPES: dict[str, str] = {"<": "\\u003c", ">": "\\u003e", "&": "\\u0026"}
REPORTED_PARAM_KEYS: tuple[str, ...] = (
    "backend",
    "model_id",
    "adapter_path",
    "gpu_name",
    "precision",
    "dtype",
    "quantization",
    "max_new_tokens",
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "num_samples",
    "seed",
    "batch_size",
)


class ReportError(ValueError):
    pass


@dataclass(frozen=True)
class ScoredRun:
    name: str
    records: list[RunRecord]
    scored: list[ScoredProblem]
    marker: str
    use_tool: bool

    @property
    def split(self) -> str:
        return self.records[0].split


def score_run_file(config: ProjectConfig, path: Path, *, tool_text_fallback: bool = True) -> ScoredRun:
    records = load_run_file(path)
    template = load_prompt(config.prompts.directory, records[0].prompt_name)
    options = scoring_options(config, template, tool_text_fallback=tool_text_fallback)
    return ScoredRun(
        name=path.stem,
        records=records,
        scored=score_records(records, options),
        marker=options.marker,
        use_tool=options.use_tool,
    )


def run_labels(names: Sequence[str]) -> list[str]:
    if len(names) < 2:
        return list(names)
    segments = [name.split(NAME_SEPARATOR) for name in names]
    shared = set(segments[0]).intersection(*segments[1:])
    labels = [NAME_SEPARATOR.join(part for part in parts if part not in shared) for parts in segments]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        return list(names)
    return labels


def default_report_path(reports_dir: Path, names: Sequence[str]) -> Path:
    return reports_dir / f"{COMPARISON_SEPARATOR.join(names)}{REPORT_SUFFIX}"


def build_report(
    config: ProjectConfig,
    runs: Sequence[ScoredRun],
    *,
    max_completion_chars: int | None = None,
) -> dict[str, Any]:
    _require_comparable(runs)
    if max_completion_chars is not None and max_completion_chars < 1:
        raise ReportError(f"max_completion_chars must be >= 1, got {max_completion_chars}")
    split = runs[0].split
    problems = {problem.id: problem for problem in load_problems(config, split)}
    _require_known_ids(runs, problems)
    references = (
        load_val_references(config.paths.train_csv, config.paths.split_file) if split == VAL_SPLIT else {}
    )
    labels = run_labels([run.name for run in runs])
    results = [_results_by_id(run, references, max_completion_chars) for run in runs]
    ordered_ids = [row_id for row_id in problems if any(row_id in by_id for by_id in results)]
    return {
        "title": " vs ".join(labels),
        "split": split,
        "has_gold": split == VAL_SPLIT,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "runs": [_run_summary(run, label, references) for run, label in zip(runs, labels)],
        "comparison": _comparison(results) if len(runs) > 1 else None,
        "problems": [_problem_row(problems[row_id], references, results, row_id) for row_id in ordered_ids],
    }


def render_report_html(report: Mapping[str, Any], template: str) -> str:
    for placeholder in (DATA_PLACEHOLDER, TITLE_PLACEHOLDER):
        if template.count(placeholder) != 1:
            raise ReportError(f"Report template must contain {placeholder} exactly once")
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    for character, escape in JSON_HTML_ESCAPES.items():
        payload = payload.replace(character, escape)
    title = html.escape(str(report["title"]))
    return template.replace(TITLE_PLACEHOLDER, title).replace(DATA_PLACEHOLDER, payload)


def write_report(path: Path, report: Mapping[str, Any], template_path: Path = TEMPLATE_PATH) -> None:
    page = render_report_html(report, template_path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def _require_comparable(runs: Sequence[ScoredRun]) -> None:
    if not runs:
        raise ReportError("At least one run is required")
    names = [run.name for run in runs]
    if len(set(names)) != len(names):
        raise ReportError(f"Run names must be unique, got {names}")
    splits = sorted({run.split for run in runs})
    if len(splits) > 1:
        raise ReportError(f"All compared runs must use the same split, got {splits}")


def _require_known_ids(runs: Sequence[ScoredRun], problems: Mapping[str, Problem]) -> None:
    for run in runs:
        unknown = [record.id for record in run.records if record.id not in problems]
        if unknown:
            raise ReportError(
                f"Run '{run.name}' has {len(unknown)} ids outside the {run.split} split, e.g. '{unknown[0]}'"
            )


def _run_summary(run: ScoredRun, label: str, references: Mapping[str, Reference]) -> dict[str, Any]:
    params = run.records[0].generation_params
    health = summarize_health(record.health_checks() for record in run.records)
    tokens = [stats.num_tokens for record in run.records for stats in record.completion_stats]
    summary: dict[str, Any] = {
        "name": run.name,
        "label": label,
        "model_id": run.records[0].model_id,
        "prompt_name": run.records[0].prompt_name,
        "marker": run.marker,
        "use_tool": run.use_tool,
        "params": {key: params.get(key) for key in REPORTED_PARAM_KEYS},
        "problems": len(run.records),
        "completions": health.completions,
        "garbage_completions": health.garbage_completions,
        "issues": health.issue_counts,
        "tool_statuses": tool_status_counts(run.scored),
        "mean_agreement": sum(problem.vote.agreement for problem in run.scored) / len(run.scored),
        "unparsed_predictions": sum(normalize_answer(problem.prediction) is None for problem in run.scored),
        "method_counts": dict(sorted(Counter(problem.method for problem in run.scored).items())),
        "tokens": _token_summary(tokens, params.get("max_new_tokens")),
        "metrics": None,
    }
    if references:
        report = evaluate_scored(run.scored, references)
        summary["metrics"] = metrics_payload(
            report,
            run_name=run.name,
            val_rows=len(references),
            health=health,
            tool_statuses=summary["tool_statuses"],
            scored=run.scored,
        )
    return summary


def _token_summary(tokens: Sequence[int], max_new_tokens: Any) -> dict[str, Any]:
    upper = max(int(max_new_tokens or 0), max(tokens), 1)
    width = max(1, -(-upper // TOKEN_HISTOGRAM_BINS))
    counts = Counter(min(count // width, TOKEN_HISTOGRAM_BINS - 1) for count in tokens)
    bins = [
        {"start": index * width, "end": (index + 1) * width, "count": counts.get(index, 0)}
        for index in range(TOKEN_HISTOGRAM_BINS)
    ]
    return {
        "mean": sum(tokens) / len(tokens),
        "max": max(tokens),
        "limit": max_new_tokens,
        "histogram": bins,
    }


def _results_by_id(
    run: ScoredRun,
    references: Mapping[str, Reference],
    max_chars: int | None,
) -> dict[str, dict[str, Any]]:
    records = {record.id: record for record in run.records}
    return {
        problem.id: _problem_result(problem, records[problem.id], references, run.use_tool, max_chars)
        for problem in run.scored
    }


def _problem_result(
    problem: ScoredProblem,
    record: RunRecord,
    references: Mapping[str, Reference],
    use_tool: bool,
    max_chars: int | None,
) -> dict[str, Any]:
    reference = references.get(problem.id)
    samples = [
        {
            "answer": sample.answer,
            "method": sample.method,
            "tool_status": sample.tool_status,
            "tool_detail": sample.tool_detail,
            "code": extract_python_code(completion) if use_tool else None,
            "tokens": stats.num_tokens,
            "finish_reason": stats.finish_reason.value,
            "health": sorted(issue.value for issue in issues),
            "garbage": is_garbage(issues),
            "completion": _clip(completion, max_chars),
        }
        for sample, completion, stats, issues in zip(
            problem.samples,
            record.completions,
            record.completion_stats,
            (assess_completion(check) for check in record.health_checks()),
        )
    ]
    return {
        "prediction": problem.prediction,
        "correct": None if reference is None else is_exact_match(problem.prediction, reference.answer),
        "method": problem.method,
        "votes": problem.vote.votes,
        "num_samples": problem.vote.num_samples,
        "distribution": [list(item) for item in problem.vote.distribution],
        "samples": samples,
        "prompt_text": _clip(record.prompt_text, max_chars),
    }


def _problem_row(
    problem: Problem,
    references: Mapping[str, Reference],
    results: Sequence[Mapping[str, dict[str, Any]]],
    row_id: str,
) -> dict[str, Any]:
    reference = references.get(row_id)
    return {
        "id": row_id,
        "skill": problem.skill,
        "skill_title": problem.skill_title,
        "statement": problem.statement,
        "gold": None if reference is None else reference.answer,
        "results": [by_id.get(row_id) for by_id in results],
    }


def _comparison(results: Sequence[Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    common = sorted(set(results[0]).intersection(*results[1:]))
    rows = [[by_id[row_id] for by_id in results] for row_id in common]
    comparison: dict[str, Any] = {
        "common_problems": len(common),
        "answers_differ": sum(len({result["prediction"] for result in row}) > 1 for row in rows),
    }
    outcomes = [[result["correct"] for result in row] for row in rows]
    if not outcomes or outcomes[0][0] is None:
        return comparison
    comparison.update(
        all_correct=sum(all(row) for row in outcomes),
        none_correct=sum(not any(row) for row in outcomes),
        common_accuracy=[sum(row[index] for row in outcomes) / len(outcomes) for index in range(len(results))],
        unique_correct=[
            sum(row[index] and sum(row) == 1 for row in outcomes) for index in range(len(results))
        ],
    )
    return comparison


def _clip(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars] + OMITTED_NOTICE.format(omitted=len(text) - max_chars)
