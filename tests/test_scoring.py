from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from conftest import MiniProject
from math_solver.generation import MockBackend, SamplingConfig
from math_solver.inference import (
    RunSettings,
    build_few_shot_context,
    load_problems,
    render_problem_messages,
    run_generation,
)
from math_solver.prompts import load_prompt
from math_solver.run_records import RunRecord, RunRecordError, read_records
from math_solver.scoring import (
    TOOL_METHOD,
    ScoredProblem,
    ScoringOptions,
    evaluate_scored,
    load_val_references,
    mispredicted_rows,
    score_records,
    tool_status_counts,
    validate_run,
)

MODEL_ID = "google/gemma-3-4b-it"
STATEMENT_PATTERN = re.compile(r"Задача: (.+)")
GREEDY = SamplingConfig(max_new_tokens=64, temperature=0.0, top_p=1.0, top_k=0, num_samples=1)
SAMPLED = SamplingConfig(max_new_tokens=64, temperature=0.7, top_p=1.0, top_k=0, num_samples=3)

Responder = Callable[[str, int], str]


def _gold(project: MiniProject, prompt: str) -> str:
    match = STATEMENT_PATTERN.search(prompt)
    assert match is not None
    return project.gold_by_statement()[match.group(1)]


def _generate(
    project: MiniProject,
    tmp_path: Path,
    responder: Responder,
    *,
    prompt: str = "baseline_cot",
    sampling: SamplingConfig = GREEDY,
) -> list[RunRecord]:
    config = project.config
    template = load_prompt(config.prompts.directory, prompt)
    problems = load_problems(config, "val")
    few_shot = build_few_shot_context(config, template, config.seed)
    messages = {problem.id: render_problem_messages(template, problem, few_shot) for problem in problems}
    settings = RunSettings(split="val", prompt_name=prompt, sampling=sampling, batch_size=4, seed=config.seed)
    output = tmp_path / f"{prompt}.jsonl"
    run_generation(problems, messages, MockBackend(model_id=MODEL_ID, responder=responder), settings, output)
    return read_records(output).records


def _options(project: MiniProject, *, use_tool: bool = False, tool_text_fallback: bool = True) -> ScoringOptions:
    config = project.config
    return ScoringOptions(
        marker="Відповідь:",
        use_tool=use_tool,
        last_number_fallback=config.extraction.last_number_fallback,
        tool_text_fallback=tool_text_fallback,
        sandbox=config.sandbox,
    )


def _references(project: MiniProject) -> dict:
    config = project.config
    return load_val_references(config.paths.train_csv, config.paths.split_file)


def _val_statements(project: MiniProject) -> list[str]:
    return [problem.statement for problem in load_problems(project.config, "val")]


def _by_id(scored: list[ScoredProblem]) -> dict[str, ScoredProblem]:
    return {problem.id: problem for problem in scored}


def test_val_accuracy_and_mispredictions(mini_project: MiniProject, tmp_path: Path) -> None:
    wrong_statement = _val_statements(mini_project)[0]

    def responder(prompt: str, index: int) -> str:
        gold = _gold(mini_project, prompt)
        return "Міркування.\nВідповідь: 999" if wrong_statement in prompt else f"Міркування.\nВідповідь: {gold}"

    records = _generate(mini_project, tmp_path, responder)
    scored = score_records(records, _options(mini_project))
    references = _references(mini_project)
    report = evaluate_scored(scored, references)
    wrong_ids = {record.id for record in records if wrong_statement in record.prompt_text}
    assert report.total == 9
    assert report.correct == 8
    assert set(report.per_skill["skill"]) == {"A-ADD", "B-MUL", "C-FRAC"}
    rows = mispredicted_rows(scored, references)
    assert {row["id"] for row in rows} == wrong_ids
    assert rows[0]["prediction"] == "999" and rows[0]["method"] == "marker"
    assert rows[0]["reference"] == references[rows[0]["id"]].answer
    assert "Відповідь: 999" in rows[0]["completion"]


def test_unknown_val_ids_are_rejected(mini_project: MiniProject, tmp_path: Path) -> None:
    records = _generate(mini_project, tmp_path, lambda prompt, index: "Відповідь: 1")
    scored = score_records(records, _options(mini_project))
    references = _references(mini_project)
    references.pop(scored[0].id)
    with pytest.raises(RunRecordError, match="not in the validation split"):
        evaluate_scored(scored, references)


def test_tool_use_scoring_prefers_sandbox_result(mini_project: MiniProject, tmp_path: Path) -> None:
    error_statement, no_code_statement = _val_statements(mini_project)[:2]

    def responder(prompt: str, index: int) -> str:
        gold = _gold(mini_project, prompt)
        if error_statement in prompt:
            return f"Код:\n```python\nanswer = 1 / 0\n```\nВідповідь: {gold}"
        if no_code_statement in prompt:
            return "Без коду, без відповіді."
        return f"Код:\n```python\nanswer = {gold} * 2 // 2\n```\nВідповідь: 999"

    records = _generate(mini_project, tmp_path, responder, prompt="tool_use_python")
    scored = _by_id(score_records(records, _options(mini_project, use_tool=True)))
    references = _references(mini_project)
    for record in records:
        problem = scored[record.id]
        gold = references[record.id].answer
        if error_statement in record.prompt_text:
            assert (problem.prediction, problem.method, problem.samples[0].tool_status) == (gold, "marker", "error")
        elif no_code_statement in record.prompt_text:
            assert (problem.prediction, problem.method, problem.samples[0].tool_status) == (None, "fallback", "no_code")
        else:
            assert (problem.prediction, problem.method) == (gold, TOOL_METHOD)
    counts = tool_status_counts(list(scored.values()))
    assert set(counts) == {"ok", "error", "no_code"} and sum(counts.values()) == 9
    report = evaluate_scored(list(scored.values()), references)
    assert TOOL_METHOD in set(report.per_method["method"])


def test_tool_use_without_text_fallback(mini_project: MiniProject, tmp_path: Path) -> None:
    def responder(prompt: str, index: int) -> str:
        return f"```python\nanswer = 1 / 0\n```\nВідповідь: {_gold(mini_project, prompt)}"

    records = _generate(mini_project, tmp_path, responder, prompt="tool_use_python")
    scored = score_records(records, _options(mini_project, use_tool=True, tool_text_fallback=False))
    assert all(problem.prediction is None and problem.method == "fallback" for problem in scored)


def test_majority_vote_over_samples(mini_project: MiniProject, tmp_path: Path) -> None:
    def responder(prompt: str, index: int) -> str:
        gold = _gold(mini_project, prompt)
        return f"Відповідь: {gold}" if index != 1 else "Відповідь: 999"

    records = _generate(mini_project, tmp_path, responder, prompt="self_consistency_cot", sampling=SAMPLED)
    scored = score_records(records, _options(mini_project))
    references = _references(mini_project)
    assert all(problem.vote.num_samples == 3 and problem.vote.votes == 2 for problem in scored)
    assert evaluate_scored(scored, references).accuracy == 1.0
    assert all(problem.vote.distribution[1] == ("999", 1) for problem in scored)


def test_tool_votes_combine_with_text_answers(mini_project: MiniProject, tmp_path: Path) -> None:
    def responder(prompt: str, index: int) -> str:
        gold = _gold(mini_project, prompt)
        if index == 0:
            return f"```python\nanswer = {gold}\n```"
        return f"```python\nanswer = (\n```\nВідповідь: {gold}" if index == 1 else "Відповідь: 999"

    records = _generate(mini_project, tmp_path, responder, prompt="tool_use_python", sampling=SAMPLED)
    scored = score_records(records, _options(mini_project, use_tool=True))
    assert all(problem.vote.votes == 2 and problem.method == TOOL_METHOD for problem in scored)
    statuses = [[sample.tool_status for sample in problem.samples] for problem in scored]
    assert all(found == ["ok", "rejected", "no_code"] for found in statuses)


def test_validate_run_rejects_mixed_runs(mini_project: MiniProject, tmp_path: Path) -> None:
    records = _generate(mini_project, tmp_path, lambda prompt, index: "Відповідь: 1")
    validate_run(records)
    other = _generate(mini_project, tmp_path, lambda prompt, index: "Відповідь: 1", prompt="plan_then_solve")
    with pytest.raises(RunRecordError, match="prompt_name"):
        validate_run([records[0], other[1]])
    with pytest.raises(RunRecordError, match="no records"):
        validate_run([])
