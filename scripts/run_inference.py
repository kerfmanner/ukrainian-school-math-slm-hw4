from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from math_solver.config import ConfigError, ProjectConfig, load_config
from math_solver.few_shot import FewShotError
from math_solver.generation import BACKENDS, GenerationError, Precision, create_backend
from math_solver.inference import (
    SPLITS,
    RunSettings,
    RunSummary,
    build_few_shot_context,
    default_run_path,
    load_problems,
    render_problem_messages,
    resolve_model,
    resolve_sampling,
    run_generation,
)
from math_solver.logging_utils import configure_logging
from math_solver.prompts import PromptError, load_prompt
from math_solver.run_records import RunRecordError
from math_solver.split import SplitError

DEFAULT_CONFIG_PATH = Path("configs/base.yaml")

logger = logging.getLogger("run_inference")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw completions into a resumable JSONL run file.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--backend", choices=BACKENDS, default=None)
    parser.add_argument("--model", default=None, help="Model key from the config or an allowed model id")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--precision", choices=[precision.value for precision in Precision], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def run(config: ProjectConfig, args: argparse.Namespace) -> RunSummary:
    model_key, model_id = resolve_model(config.models, args.model)
    prompt_name = args.prompt or config.prompts.default
    template = load_prompt(config.prompts.directory, prompt_name)
    seed = config.seed if args.seed is None else args.seed
    settings = RunSettings(
        split=args.split,
        prompt_name=prompt_name,
        sampling=resolve_sampling(
            template,
            config.inference,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        ),
        batch_size=args.batch_size or config.inference.batch_size,
        seed=seed,
    )
    problems = load_problems(config, args.split, args.limit)
    few_shot = build_few_shot_context(config, template, seed)
    messages = {problem.id: render_problem_messages(template, problem, few_shot) for problem in problems}
    output = args.output or default_run_path(
        config.paths.outputs_dir,
        model_key=model_key,
        prompt_name=prompt_name,
        split=args.split,
        adapter_path=args.adapter_path,
        tag=args.run_tag,
    )
    logger.info("model=%s prompt=%s split=%s sampling=%s", model_id, prompt_name, args.split, settings.sampling)
    backend = create_backend(
        args.backend or config.inference.backend,
        model_id=model_id,
        adapter_path=args.adapter_path,
        precision=Precision(args.precision or config.inference.precision),
    )
    logger.info("backend=%s", backend.info)
    return run_generation(problems, messages, backend, settings, output)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    try:
        summary = run(load_config(args.config), args)
    except (
        ConfigError,
        PromptError,
        FewShotError,
        SplitError,
        RunRecordError,
        GenerationError,
        FileNotFoundError,
        ValueError,
    ) as error:
        logger.error("%s", error)
        return 1
    logger.info(
        "done: %s total=%d resumed=%d generated=%d",
        summary.output_path,
        summary.total,
        summary.resumed,
        summary.generated,
    )
    logger.info("%s", summary.health.format())
    if summary.health.garbage_completions:
        logger.warning("Found %d garbage completions; inspect them before scoring", summary.health.garbage_completions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
