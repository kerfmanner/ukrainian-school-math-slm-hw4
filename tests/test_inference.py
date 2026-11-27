from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import MiniProject
from math_solver.few_shot import FewShotError
from math_solver.generation import GenerationError, MockBackend, SamplingConfig
from math_solver.inference import (
    Problem,
    RunSettings,
    build_few_shot_context,
    default_run_path,
    load_problems,
    render_problem_messages,
    resolve_model,
    resolve_sampling,
    run_generation,
)
from math_solver.prompts import PromptTemplate, load_prompt
from math_solver.run_records import RunRecordError, read_records
from math_solver.structured import ConfigError

MODEL_ID = "google/gemma-3-4b-it"
GREEDY = SamplingConfig(max_new_tokens=64, temperature=0.0, top_p=1.0, top_k=0, num_samples=1)


def _settings(sampling: SamplingConfig = GREEDY, batch_size: int = 4, prompt: str = "baseline_cot") -> RunSettings:
    return RunSettings(split="val", prompt_name=prompt, sampling=sampling, batch_size=batch_size, seed=42)


def _prepare(
    project: MiniProject, prompt: str = "baseline_cot", limit: int | None = None
) -> tuple[list[Problem], dict]:
    config = project.config
    template = load_prompt(config.prompts.directory, prompt)
    problems = load_problems(config, "val", limit)
    few_shot = build_few_shot_context(config, template, config.seed)
    return problems, {problem.id: render_problem_messages(template, problem, few_shot) for problem in problems}


def _template(project: MiniProject, name: str) -> PromptTemplate:
    return load_prompt(project.config.prompts.directory, name)


def test_batches_cover_all_problems_in_order(mini_project: MiniProject, tmp_path: Path) -> None:
    problems, messages = _prepare(mini_project)
    backend = MockBackend(model_id=MODEL_ID)
    output = tmp_path / "run.jsonl"
    summary = run_generation(problems, messages, backend, _settings(batch_size=4), output)
    assert len(problems) == 9
    assert backend.batch_sizes == [4, 4, 1]
    records = read_records(output).records
    assert [record.id for record in records] == [problem.id for problem in problems]
    assert summary.generated == 9 and summary.resumed == 0
    assert summary.health.garbage_completions == 0


def test_record_contains_prompt_and_generation_params(mini_project: MiniProject, tmp_path: Path) -> None:
    problems, messages = _prepare(mini_project, limit=1)
    output = tmp_path / "run.jsonl"
    run_generation(problems, messages, MockBackend(model_id=MODEL_ID), _settings(), output)
    payload = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert {"id", "skill", "model_id", "prompt_name", "prompt_text", "completions", "generation_params"} <= set(payload)
    assert problems[0].statement in payload["prompt_text"]
    params = payload["generation_params"]
    assert {"gpu_name", "dtype", "quantization", "precision", "batch_seed", "seed"} <= set(params)
    assert params["model_id"] == MODEL_ID and params["num_samples"] == 1


def test_limit_takes_first_rows_of_fixed_split(mini_project: MiniProject) -> None:
    full = load_problems(mini_project.config, "val")
    assert load_problems(mini_project.config, "val", limit=3) == full[:3]
    with pytest.raises(ConfigError, match="limit"):
        load_problems(mini_project.config, "val", limit=0)


def test_test_split_loads_test_csv(mini_project: MiniProject) -> None:
    problems = load_problems(mini_project.config, "test")
    assert [problem.id for problem in problems] == ["t0", "t1", "t2", "t3", "t4"]


def test_resume_skips_completed_ids(mini_project: MiniProject, tmp_path: Path) -> None:
    problems, messages = _prepare(mini_project)
    output = tmp_path / "run.jsonl"
    run_generation(problems[:4], messages, MockBackend(model_id=MODEL_ID), _settings(), output)
    first_lines = output.read_text(encoding="utf-8").splitlines()
    backend = MockBackend(model_id=MODEL_ID)
    summary = run_generation(problems, messages, backend, _settings(), output)
    assert sum(backend.batch_sizes) == 5
    assert summary.resumed == 4 and summary.generated == 5
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[:4] == first_lines
    assert sorted(record.id for record in read_records(output).records) == sorted(problem.id for problem in problems)


def test_finished_run_generates_nothing(mini_project: MiniProject, tmp_path: Path) -> None:
    problems, messages = _prepare(mini_project)
    output = tmp_path / "run.jsonl"
    run_generation(problems, messages, MockBackend(model_id=MODEL_ID), _settings(), output)
    backend = MockBackend(model_id=MODEL_ID)
    summary = run_generation(problems, messages, backend, _settings(), output)
    assert backend.batch_sizes == [] and summary.generated == 0


def test_resume_repairs_partially_written_last_line(mini_project: MiniProject, tmp_path: Path) -> None:
    problems, messages = _prepare(mini_project)
    output = tmp_path / "run.jsonl"
    run_generation(problems[:2], messages, MockBackend(model_id=MODEL_ID), _settings(), output)
    with output.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "' + problems[2].id + '", "completions": ["обірва')
    run_generation(problems[:3], messages, MockBackend(model_id=MODEL_ID), _settings(), output)
    assert [record.id for record in read_records(output).records] == [problem.id for problem in problems[:3]]
    assert read_records(output).truncated_tail_offset is None


def test_resume_with_different_settings_raises(mini_project: MiniProject, tmp_path: Path) -> None:
    problems, messages = _prepare(mini_project)
    output = tmp_path / "run.jsonl"
    run_generation(problems[:2], messages, MockBackend(model_id=MODEL_ID), _settings(), output)
    with pytest.raises(RunRecordError, match="max_new_tokens"):
        changed = _settings(replace(GREEDY, max_new_tokens=8))
        run_generation(problems, messages, MockBackend(model_id=MODEL_ID), changed, output)


def test_resume_with_changed_prompt_text_raises(mini_project: MiniProject, tmp_path: Path) -> None:
    problems, messages = _prepare(mini_project)
    output = tmp_path / "run.jsonl"
    run_generation(problems[:2], messages, MockBackend(model_id=MODEL_ID), _settings(), output)
    changed = {**messages, problems[0].id: [{"role": "user", "content": "інший текст"}]}
    with pytest.raises(RunRecordError, match="different prompt text"):
        run_generation(problems, changed, MockBackend(model_id=MODEL_ID), _settings(), output)


def test_multi_sample_records(mini_project: MiniProject, tmp_path: Path) -> None:
    problems, messages = _prepare(mini_project, limit=3)
    sampling = SamplingConfig(max_new_tokens=64, temperature=0.7, top_p=0.95, top_k=0, num_samples=3)
    output = tmp_path / "run.jsonl"
    backend = MockBackend(model_id=MODEL_ID)
    run_generation(problems, messages, backend, _settings(sampling, batch_size=2), output)
    records = read_records(output).records
    assert backend.batch_sizes == [2, 1]
    assert all(len(record.completions) == 3 and len(record.completion_stats) == 3 for record in records)
    assert records[0].completions[0] != records[0].completions[1]


def test_first_batch_of_garbage_aborts_without_writing(mini_project: MiniProject, tmp_path: Path) -> None:
    problems, messages = _prepare(mini_project)
    output = tmp_path / "run.jsonl"
    backend = MockBackend(model_id=MODEL_ID, responder=lambda prompt, index: "")
    with pytest.raises(GenerationError, match="first batch"):
        run_generation(problems, messages, backend, _settings(), output)
    assert not output.exists()


def test_garbage_after_first_batch_is_counted(mini_project: MiniProject, tmp_path: Path) -> None:
    problems, messages = _prepare(mini_project)
    calls = iter(range(len(problems)))

    def responder(prompt: str, index: int) -> str:
        return "Відповідь: 1" if next(calls) < 4 else "漢字ประเทศไทยქართული" * 5

    backend = MockBackend(model_id=MODEL_ID, responder=responder)
    summary = run_generation(problems, messages, backend, _settings(batch_size=4), tmp_path / "run.jsonl")
    assert summary.health.garbage_completions == 5
    assert summary.health.issue_counts["garbled"] == 5


def test_resolve_sampling_uses_self_consistency_spec(mini_project: MiniProject) -> None:
    defaults = mini_project.config.inference
    sampling = resolve_sampling(_template(mini_project, "self_consistency_cot"), defaults)
    assert (sampling.num_samples, sampling.temperature) == (5, 0.7)
    assert resolve_sampling(_template(mini_project, "self_consistency_cot"), defaults, num_samples=3).num_samples == 3
    with pytest.raises(ConfigError, match="self-consistency"):
        resolve_sampling(_template(mini_project, "self_consistency_cot"), defaults, num_samples=1)


def test_resolve_sampling_defaults_and_overrides(mini_project: MiniProject) -> None:
    defaults = mini_project.config.inference
    template = _template(mini_project, "baseline_cot")
    assert resolve_sampling(template, defaults).num_samples == 1
    assert resolve_sampling(template, defaults, max_new_tokens=32).max_new_tokens == 32


def test_resolve_sampling_prefers_prompt_token_cap_over_defaults(mini_project: MiniProject) -> None:
    defaults = mini_project.config.inference
    template = _template(mini_project, "few_shot_direct")
    assert template.max_new_tokens is not None
    assert resolve_sampling(template, defaults).max_new_tokens == template.max_new_tokens
    assert resolve_sampling(template, defaults, max_new_tokens=16).max_new_tokens == 16
    with pytest.raises(ConfigError, match="temperature > 0"):
        resolve_sampling(template, defaults, num_samples=4)


def test_resolve_model_accepts_keys_and_ids(mini_project: MiniProject) -> None:
    models = mini_project.config.models
    assert resolve_model(models, None) == ("gemma3_4b_it", MODEL_ID)
    assert resolve_model(models, "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0")[0] == "mamaylm_gemma3_4b_it"
    with pytest.raises(ConfigError, match="Unknown model"):
        resolve_model(models, "Qwen/Qwen3-8B")


def test_default_run_path_is_descriptive(tmp_path: Path) -> None:
    path = default_run_path(
        tmp_path,
        model_key="gemma3_4b_it",
        prompt_name="baseline_cot",
        split="val",
        adapter_path=Path("ckpt/a1"),
        tag="t4",
    )
    assert path == tmp_path / "runs" / "gemma3_4b_it__baseline_cot__val__lora-a1__t4.jsonl"


def test_few_shot_direct_uses_train_examples_only(mini_project: MiniProject) -> None:
    problems, messages = _prepare(mini_project, prompt="few_shot_direct")
    val_statements = {problem.statement for problem in problems}
    turns = messages[problems[0].id]
    example_questions = [turn["content"] for turn in turns[:-1] if turn["role"] == "user"]
    shown = {match for content in example_questions for match in re.findall(r"Задача: (.+)", content)}
    assert shown and not shown & val_statements
    assert [turn["role"] for turn in turns[1:]] == ["user", "assistant"] * len(example_questions) + ["user"]


def test_few_shot_cot_without_solutions_fails_fast(mini_project: MiniProject) -> None:
    with pytest.raises(FewShotError, match="with a solution"):
        _prepare(mini_project, prompt="few_shot_cot")
