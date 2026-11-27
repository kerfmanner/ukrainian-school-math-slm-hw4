from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from math_solver.config import ConfigError, load_config
from math_solver.logging_utils import configure_logging
from math_solver.prompts import PromptError
from math_solver.run_records import RunRecordError
from math_solver.run_report import (
    TEMPLATE_PATH,
    ReportError,
    build_report,
    default_report_path,
    score_run_file,
    write_report,
)
from math_solver.split import SplitError

DEFAULT_CONFIG_PATH = Path("configs/base.yaml")
REPORT_VIEWS_SUBDIR = "report_views"

logger = logging.getLogger("render_run_report")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one run, or a comparison of several runs on the same split, as a self-contained HTML page."
    )
    parser.add_argument("--run", type=Path, action="append", required=True, help="Run JSONL; repeat to compare runs")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-completion-chars", type=int, default=None)
    parser.add_argument("--tool-text-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    runs = [score_run_file(config, path, tool_text_fallback=args.tool_text_fallback) for path in args.run]
    report = build_report(config, runs, max_completion_chars=args.max_completion_chars)
    output = args.output or default_report_path(config.paths.outputs_dir / REPORT_VIEWS_SUBDIR, [item.name for item in runs])
    write_report(output, report, args.template)
    for summary in report["runs"]:
        metrics = summary["metrics"]
        accuracy = "n/a" if metrics is None else f"{metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['evaluated']})"
        logger.info("%s: problems=%d accuracy=%s", summary["label"], summary["problems"], accuracy)
    logger.info("Wrote %s (%d problems, %.1f KB)", output, len(report["problems"]), output.stat().st_size / 1024)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    try:
        run(args)
    except (ConfigError, PromptError, RunRecordError, SplitError, ReportError, FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
