from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from math_solver.config import InferenceConfig, ModelsConfig, ProjectConfig
from math_solver.data_io import read_text_csv, require_columns
from math_solver.few_shot import WORKED_SOURCE, build_few_shot_pool, load_worked_solutions, select_few_shot_examples
from math_solver.generation import (
    BackendInfo,
    Generation,
    GenerationError,
    GeneratorBackend,
    SamplingConfig,
    derive_batch_seed,
)
from math_solver.health import HealthSummary, assess_completion, is_garbage, summarize_health
from math_solver.prompts import ChatMessage, PromptTemplate
from math_solver.run_records import (
    CompletionStats,
    RunRecord,
    RunRecordError,
    append_records,
    make_signature,
    read_records,
    repair_truncated_tail,
    require_consistent_signature,
)
from math_solver.split import apply_split, load_split
from math_solver.structured import ConfigError

VAL_SPLIT = "val"
TEST_SPLIT = "test"
SPLITS: tuple[str, ...] = (VAL_SPLIT, TEST_SPLIT)
PROBLEM_COLUMNS: tuple[str, ...] = ("id", "skill", "skill_title", "statement")
RUNS_SUBDIR = "runs"
RUN_FILE_SUFFIX = ".jsonl"
NAME_SEPARATOR = "__"
ADAPTER_NAME_PREFIX = "lora-"
MAX_REPORTED_TEXT_CHARS = 200

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Problem:
    id: str
    skill: str
    skill_title: str
    statement: str


@dataclass(frozen=True)
class FewShotContext:
    pool: pd.DataFrame
    worked_solutions: Mapping[str, str]
    seed: int


@dataclass(frozen=True)
class RunSettings:
    split: str
    prompt_name: str
    sampling: SamplingConfig
    batch_size: int
    seed: int

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ConfigError(f"split must be one of {list(SPLITS)}, got '{self.split}'")
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {self.batch_size}")


@dataclass(frozen=True)
class RunSummary:
    output_path: Path
    total: int
    resumed: int
    generated: int
    health: HealthSummary


def resolve_model(models: ModelsConfig, requested: str | None) -> tuple[str, str]:
    if requested is None:
        return models.default, models.default_model_id
    if requested in models.candidates:
        return requested, models.candidates[requested]
    for key, model_id in models.candidates.items():
        if model_id == requested:
            return key, model_id
    raise ConfigError(f"Unknown model '{requested}'; use one of {sorted(models.candidates)} or their model ids")


def resolve_sampling(
    template: PromptTemplate,
    defaults: InferenceConfig,
    *,
    num_samples: int | None = None,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
) -> SamplingConfig:
    spec = template.self_consistency
    spec_samples = None if spec is None else spec.num_samples
    spec_temperature = None if spec is None else spec.temperature
    resolved_samples = _first_set(num_samples, spec_samples, defaults.num_samples)
    if spec is not None and resolved_samples < 2:
        raise ConfigError(f"Prompt '{template.name}' is a self-consistency prompt and needs num_samples >= 2")
    try:
        return SamplingConfig(
            max_new_tokens=_first_set(max_new_tokens, defaults.max_new_tokens),
            temperature=_first_set(temperature, spec_temperature, defaults.temperature),
            top_p=defaults.top_p,
            top_k=defaults.top_k,
            num_samples=resolved_samples,
        )
    except ValueError as error:
        raise ConfigError(str(error)) from error


def default_run_path(
    outputs_dir: Path,
    *,
    model_key: str,
    prompt_name: str,
    split: str,
    adapter_path: Path | None = None,
    tag: str | None = None,
) -> Path:
    parts = [model_key, prompt_name, split]
    if adapter_path is not None:
        parts.append(f"{ADAPTER_NAME_PREFIX}{adapter_path.resolve().name}")
    if tag:
        parts.append(tag)
    return outputs_dir / RUNS_SUBDIR / f"{NAME_SEPARATOR.join(parts)}{RUN_FILE_SUFFIX}"


def load_problems(config: ProjectConfig, split: str, limit: int | None = None) -> list[Problem]:
    if limit is not None and limit < 1:
        raise ConfigError(f"limit must be >= 1, got {limit}")
    if split == VAL_SPLIT:
        _, frame = apply_split(read_text_csv(config.paths.train_csv), load_split(config.paths.split_file))
    elif split == TEST_SPLIT:
        frame = read_text_csv(config.paths.test_csv)
    else:
        raise ConfigError(f"split must be one of {list(SPLITS)}, got '{split}'")
    require_columns(frame, PROBLEM_COLUMNS, f"{split} problems")
    problems = [Problem(**{column: str(row[column]) for column in PROBLEM_COLUMNS}) for row in frame.to_dict("records")]
    return problems if limit is None else problems[:limit]


def build_few_shot_context(config: ProjectConfig, template: PromptTemplate, seed: int) -> FewShotContext | None:
    if template.few_shot is None:
        return None
    pool = build_few_shot_pool(read_text_csv(config.paths.train_csv), load_split(config.paths.split_file))
    worked: Mapping[str, str] = {}
    worked_path = config.paths.worked_solutions_jsonl
    if WORKED_SOURCE in template.few_shot.solution_sources and worked_path is not None:
        worked = load_worked_solutions(worked_path, pool, marker=template.answer_marker)
    return FewShotContext(pool=pool, worked_solutions=worked, seed=seed)


def render_problem_messages(
    template: PromptTemplate,
    problem: Problem,
    few_shot: FewShotContext | None,
) -> list[ChatMessage]:
    examples: list[dict[str, str]] = []
    if few_shot is not None and template.few_shot is not None:
        examples = select_few_shot_examples(
            few_shot.pool,
            template.few_shot,
            skill=problem.skill,
            seed=few_shot.seed,
            exclude_ids={problem.id},
            worked_solutions=few_shot.worked_solutions,
        )
    return template.render_messages(statement=problem.statement, skill_title=problem.skill_title, examples=examples)


def generation_params(info: BackendInfo, settings: RunSettings) -> dict[str, Any]:
    sampling = settings.sampling
    return {
        "backend": info.backend,
        "model_id": info.model_id,
        "adapter_path": info.adapter_path,
        "precision": info.precision,
        "dtype": info.dtype,
        "quantization": info.quantization,
        "device": info.device,
        "gpu_name": info.gpu_name,
        "max_new_tokens": sampling.max_new_tokens,
        "do_sample": sampling.do_sample,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "num_samples": sampling.num_samples,
        "seed": settings.seed,
        "batch_size": settings.batch_size,
    }


def run_generation(
    problems: Sequence[Problem],
    messages: Mapping[str, Sequence[ChatMessage]],
    backend: GeneratorBackend,
    settings: RunSettings,
    output_path: Path,
) -> RunSummary:
    if not problems:
        raise ConfigError("No problems to run")
    prompts = {problem.id: backend.format_prompt(messages[problem.id]) for problem in problems}
    params = generation_params(backend.info, settings)
    expected = make_signature(settings.split, backend.info.model_id, settings.prompt_name, params)
    existing = _load_existing(output_path, expected, prompts)
    pending = [problem for problem in problems if problem.id not in existing]
    logger.info(
        "run=%s problems=%d resumed=%d pending=%d batch_size=%d seed=%d",
        output_path.name,
        len(problems),
        len(problems) - len(pending),
        len(pending),
        settings.batch_size,
        settings.seed,
    )
    generated = _generate_pending(pending, prompts, backend, settings, params, output_path, existing)
    health = summarize_health(existing[problem.id].health_checks() for problem in problems)
    return RunSummary(output_path, len(problems), len(problems) - len(pending), generated, health)


def _generate_pending(
    pending: Sequence[Problem],
    prompts: Mapping[str, str],
    backend: GeneratorBackend,
    settings: RunSettings,
    params: Mapping[str, Any],
    output_path: Path,
    completed: dict[str, RunRecord],
) -> int:
    batches = [pending[start : start + settings.batch_size] for start in range(0, len(pending), settings.batch_size)]
    for index, batch in enumerate(batches, start=1):
        ids = [problem.id for problem in batch]
        batch_seed = derive_batch_seed(settings.seed, ids)
        outputs = backend.generate([prompts[row_id] for row_id in ids], settings.sampling, batch_seed)
        _require_shape(outputs, len(batch), settings.sampling.num_samples)
        records = [
            _to_record(problem, prompts[problem.id], generations, settings, {**params, "batch_seed": batch_seed})
            for problem, generations in zip(batch, outputs)
        ]
        if index == 1:
            _abort_if_all_garbage(records)
        append_records(output_path, records)
        completed.update((record.id, record) for record in records)
        logger.info("batch %d/%d written (%d problems)", index, len(batches), len(batch))
    return sum(len(batch) for batch in batches)


def _load_existing(path: Path, expected: Mapping[str, Any], prompts: Mapping[str, str]) -> dict[str, RunRecord]:
    records_file = read_records(path)
    if records_file.truncated_tail_offset is not None:
        logger.warning("Dropping a partially written last line in %s", path)
        repair_truncated_tail(path, records_file.truncated_tail_offset)
    records = records_file.records
    require_consistent_signature(records, expected)
    for record in records:
        if record.id in prompts and record.prompt_text != prompts[record.id]:
            raise RunRecordError(
                f"Record '{record.id}' in {path} was generated from a different prompt text; "
                "use a new --run-tag or --output instead of resuming"
            )
    return {record.id: record for record in records}


def _to_record(
    problem: Problem,
    prompt_text: str,
    generations: Sequence[Generation],
    settings: RunSettings,
    params: Mapping[str, Any],
) -> RunRecord:
    return RunRecord(
        id=problem.id,
        skill=problem.skill,
        split=settings.split,
        model_id=str(params["model_id"]),
        prompt_name=settings.prompt_name,
        prompt_text=prompt_text,
        completions=tuple(generation.text for generation in generations),
        completion_stats=tuple(CompletionStats.from_generation(generation) for generation in generations),
        generation_params=dict(params),
    )


def _require_shape(outputs: Sequence[Sequence[Generation]], prompts: int, samples: int) -> None:
    if len(outputs) != prompts or any(len(group) != samples for group in outputs):
        shape = [len(group) for group in outputs]
        raise GenerationError(f"Backend returned sample counts {shape}, expected {prompts} x {samples}")


def _abort_if_all_garbage(records: Sequence[RunRecord]) -> None:
    checks = [check for record in records for check in record.health_checks()]
    issues = [assess_completion(check) for check in checks]
    if issues and all(is_garbage(found) for found in issues):
        example = records[0].completions[0][:MAX_REPORTED_TEXT_CHARS]
        flagged = sorted({issue.value for found in issues for issue in found})
        raise GenerationError(
            f"Every completion in the first batch is empty/garbled/repetitive ({flagged}); aborting before writing. "
            f"Check precision/dtype. First completion: {example!r}"
        )


def _first_set(*values: Any) -> Any:
    return next(value for value in values if value is not None)
