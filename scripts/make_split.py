from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from math_solver.config import ConfigError, ProjectConfig, load_config
from math_solver.data_io import read_text_csv
from math_solver.logging_utils import configure_logging
from math_solver.split import SplitError, load_split, make_stratified_split, save_split

DEFAULT_CONFIG_PATH = Path("configs/base.yaml")

logger = logging.getLogger("make_split")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the fixed skill-stratified train/validation split.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def apply_overrides(config: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    split = config.split if args.val_ratio is None else replace(config.split, val_ratio=args.val_ratio)
    paths = config.paths
    if args.train_csv is not None:
        paths = replace(paths, train_csv=args.train_csv)
    if args.output is not None:
        paths = replace(paths, split_file=args.output)
    seed = config.seed if args.seed is None else args.seed
    return replace(config, seed=seed, split=split, paths=paths)


def run(config: ProjectConfig, overwrite: bool) -> int:
    frame = read_text_csv(config.paths.train_csv)
    split = make_stratified_split(
        frame,
        val_ratio=config.split.val_ratio,
        seed=config.seed,
        stratify_column=config.split.stratify_column,
    )
    logger.info(
        "seed=%d val_ratio=%.3f strata=%d train=%d val=%d",
        split.seed,
        split.val_ratio,
        frame[split.stratify_column].nunique(),
        len(split.train_ids),
        len(split.val_ids),
    )
    output = config.paths.split_file
    if output.exists() and not overwrite:
        if load_split(output) == split:
            logger.info("Existing split at %s is identical; nothing to do", output)
            return 0
        logger.error("Split at %s differs from the new one; pass --overwrite to replace it", output)
        return 1
    save_split(split, output)
    logger.info("Saved split to %s", output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = apply_overrides(load_config(args.config), args)
        return run(config, args.overwrite)
    except (ConfigError, SplitError, FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
