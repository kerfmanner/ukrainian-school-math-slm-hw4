from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from math_solver.config import ConfigError, load_config
from math_solver.data_io import read_text_csv
from math_solver.logging_utils import configure_logging
from math_solver.submission import SubmissionCheck, SubmissionError, check_submission

DEFAULT_CONFIG_PATH = Path("configs/base.yaml")

logger = logging.getLogger("validate_submission")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a submission CSV against sample_submission.csv.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--sample", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--fallback-answer", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def report(check: SubmissionCheck, fallback_answer: str) -> None:
    fallback_share = check.fallback_count / check.rows if check.rows else 0.0
    logger.info("rows=%d expected=%d", check.rows, check.expected_rows)
    logger.info("missing=%d non_canonical=%d", check.missing_count, check.non_canonical_count)
    logger.info("fallback('%s')=%d (%.1f%%)", fallback_answer, check.fallback_count, 100 * fallback_share)
    for problem in check.problems:
        logger.error("%s", problem)
    logger.info("VALID" if check.is_valid else "INVALID")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_config(args.config)
        sample_path = args.sample or config.paths.sample_submission_csv
        expected_rows = args.expected_rows or config.submission.expected_rows
        fallback_answer = args.fallback_answer or config.submission.fallback_answer
        check = check_submission(
            read_text_csv(args.submission),
            read_text_csv(sample_path),
            expected_rows=expected_rows,
            fallback_answer=fallback_answer,
        )
    except (ConfigError, SubmissionError, FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1
    report(check, fallback_answer)
    return 0 if check.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
