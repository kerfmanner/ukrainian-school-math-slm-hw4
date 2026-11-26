from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from math_solver.config import ALLOWED_SUBMISSION_MODEL_IDS, ConfigError, ProjectConfig, load_config
from math_solver.data_io import read_text_csv
from math_solver.generation import MOCK_BACKEND
from math_solver.health import summarize_health
from math_solver.inference import TEST_SPLIT, VAL_SPLIT
from math_solver.logging_utils import configure_logging
from math_solver.prompts import PromptError, load_prompt
from math_solver.run_records import RunRecord, RunRecordError, read_records
from math_solver.scoring import (
    ScoredProblem,
    ScoringOptions,
    evaluate_scored,
    load_val_references,
    metrics_payload,
    mispredicted_rows,
    score_records,
    tool_status_counts,
    validate_run,
    write_jsonl,
)
from math_solver.split import SplitError
from math_solver.submission import ID_COLUMN, SubmissionError, build_submission, save_submission

DEFAULT_CONFIG_PATH = Path("configs/base.yaml")
VALIDATOR_PATH = Path(__file__).with_name("validate_submission.py")
REPORTS_SUBDIR = "reports"
SUBMISSIONS_SUBDIR = "submissions"
ERRORS_SUFFIX = "__errors.jsonl"
METRICS_SUFFIX = "__metrics.json"

logger = logging.getLogger("score_run")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a raw completions JSONL offline on CPU.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--submission", type=Path, default=None)
    parser.add_argument("--tool-text-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def load_run(path: Path) -> list[RunRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Run file not found: {path}")
    records_file = read_records(path)
    if records_file.truncated_tail_offset is not None:
        logger.warning("Ignoring a partially written last line in %s", path)
    validate_run(records_file.records)
    return records_file.records


def score_val(
    config: ProjectConfig,
    args: argparse.Namespace,
    scored: list[ScoredProblem],
    records: list[RunRecord],
) -> int:
    references = load_val_references(config.paths.train_csv, config.paths.split_file)
    report = evaluate_scored(scored, references)
    if report.total < len(references):
        logger.warning("Run covers %d of %d validation rows", report.total, len(references))
    logger.info(
        "accuracy=%.4f (%d/%d) unparsed=%d",
        report.accuracy,
        report.correct,
        report.total,
        report.unparsed_predictions,
    )
    logger.info("per-skill accuracy:\n%s", report.per_skill.to_string(index=False, float_format="%.3f"))
    if report.per_method is not None:
        logger.info("per-method accuracy:\n%s", report.per_method.to_string(index=False, float_format="%.3f"))
    report_dir = args.report_dir or config.paths.outputs_dir / REPORTS_SUBDIR
    errors_path = report_dir / f"{args.run.stem}{ERRORS_SUFFIX}"
    metrics_path = report_dir / f"{args.run.stem}{METRICS_SUFFIX}"
    errors = mispredicted_rows(scored, references)
    write_jsonl(errors_path, errors)
    payload = metrics_payload(
        report,
        run_name=args.run.stem,
        val_rows=len(references),
        health=summarize_health(record.health_checks() for record in records),
        tool_statuses=tool_status_counts(scored),
        scored=scored,
    )
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved %d mispredicted rows to %s and metrics to %s", len(errors), errors_path, metrics_path)
    return 0


def score_test(
    config: ProjectConfig,
    args: argparse.Namespace,
    scored: list[ScoredProblem],
    records: list[RunRecord],
) -> int:
    params = records[0].generation_params
    if params.get("backend") == MOCK_BACKEND and not args.allow_mock:
        raise SubmissionError("Run was produced by the mock backend; pass --allow-mock only for pipeline tests")
    if records[0].model_id not in ALLOWED_SUBMISSION_MODEL_IDS:
        raise SubmissionError(f"Model '{records[0].model_id}' is not allowed for submissions")
    sample = read_text_csv(config.paths.sample_submission_csv)
    predictions = {problem.id: problem.prediction for problem in scored}
    missing = set(sample[ID_COLUMN].astype(str)) - set(predictions)
    if missing and not args.allow_partial:
        raise SubmissionError(f"Run is missing {len(missing)} test ids; finish the run or pass --allow-partial")
    submission = build_submission(sample, predictions, config.submission.fallback_answer)
    path = args.submission or config.paths.outputs_dir / SUBMISSIONS_SUBDIR / f"{args.run.stem}.csv"
    save_submission(submission, sample, path, config.submission.expected_rows)
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), "--submission", str(path), "--config", str(args.config)],
        check=False,
    )
    return completed.returncode


def run(config: ProjectConfig, args: argparse.Namespace) -> int:
    records = load_run(args.run)
    template = load_prompt(config.prompts.directory, records[0].prompt_name)
    options = ScoringOptions(
        marker=template.answer_marker,
        use_tool=template.tool_use is not None,
        last_number_fallback=config.extraction.last_number_fallback,
        tool_text_fallback=args.tool_text_fallback,
        sandbox=config.sandbox,
    )
    split = records[0].split
    logger.info(
        "run=%s split=%s prompt=%s problems=%d tool_use=%s",
        args.run.name,
        split,
        template.name,
        len(records),
        options.use_tool,
    )
    scored = score_records(records, options)
    logger.info("%s", summarize_health(record.health_checks() for record in records).format())
    if options.use_tool:
        logger.info("tool statuses: %s", tool_status_counts(scored))
    if split == VAL_SPLIT:
        return score_val(config, args, scored, records)
    if split == TEST_SPLIT:
        return score_test(config, args, scored, records)
    raise RunRecordError(f"Unknown split '{split}' in run file")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    try:
        return run(load_config(args.config), args)
    except (
        ConfigError,
        PromptError,
        RunRecordError,
        SplitError,
        SubmissionError,
        FileNotFoundError,
        ValueError,
    ) as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
