from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from conftest import REPO_ROOT
from math_solver.few_shot import (
    FewShotError,
    build_few_shot_pool,
    load_worked_solutions,
    parse_steps,
    render_solution,
    select_few_shot_examples,
)
from math_solver.prompts import FewShotSpec, load_prompt
from math_solver.split import make_stratified_split

SEED = 42
DIRECT_SPEC = FewShotSpec(num_examples=3, same_skill=True, example="{{statement}} -> {{answer}}")
STEPS_SPEC = FewShotSpec(
    num_examples=3,
    same_skill=True,
    example="{{statement}}\n{{solution}}\n{{answer}}",
    solution_sources=("worked", "steps"),
    steps_template="Кроки: {{steps}}",
)
STRICT_SPEC = FewShotSpec(
    num_examples=3,
    same_skill=True,
    example="{{statement}}\n{{solution}}\n{{answer}}",
    solution_sources=("worked", "steps"),
    steps_template="Кроки: {{steps}}",
    require_solution=True,
)


@pytest.fixture
def train_with_steps(train_frame: pd.DataFrame) -> pd.DataFrame:
    frame = train_frame.copy()
    frame["steps"] = "[]"
    geometry = frame["skill"] == "geometry"
    frame.loc[geometry, "steps"] = [json.dumps([str(index), str(index * 2)]) for index in range(int(geometry.sum()))]
    return frame


@pytest.fixture
def split(train_with_steps: pd.DataFrame):
    return make_stratified_split(train_with_steps, val_ratio=0.2, seed=SEED, stratify_column="skill")


@pytest.fixture
def pool(train_with_steps: pd.DataFrame, split) -> pd.DataFrame:
    return build_few_shot_pool(train_with_steps, split)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")
    return path


def test_pool_contains_only_train_split_rows(train_with_steps: pd.DataFrame, split) -> None:
    pool = build_few_shot_pool(train_with_steps, split)
    assert set(pool["id"]) == set(split.train_ids)
    assert not set(pool["id"]) & set(split.val_ids)


def test_direct_selection_is_deterministic_and_same_skill(pool: pd.DataFrame) -> None:
    first = select_few_shot_examples(pool, DIRECT_SPEC, skill="percent", seed=SEED)
    reshuffled = pool.sample(frac=1, random_state=0)
    assert first == select_few_shot_examples(reshuffled, DIRECT_SPEC, skill="percent", seed=SEED)
    assert len(first) == 3
    assert {example["statement"] for example in first} <= set(pool.loc[pool["skill"] == "percent", "statement"])
    assert all(example["solution"] == "" for example in first)


def test_selection_excludes_ids(pool: pd.DataFrame) -> None:
    percent_ids = pool.loc[pool["skill"] == "percent", "id"].tolist()
    examples = select_few_shot_examples(pool, DIRECT_SPEC, skill="percent", seed=SEED, exclude_ids=percent_ids[1:])
    assert [example["statement"] for example in examples] == pool.loc[pool["id"] == percent_ids[0], "statement"].tolist()


def test_selection_caps_at_available(pool: pd.DataFrame) -> None:
    spec = FewShotSpec(num_examples=100, same_skill=True, example="{{statement}} {{answer}}")
    assert len(select_few_shot_examples(pool, spec, skill="percent", seed=SEED)) == int((pool["skill"] == "percent").sum())


def test_selection_across_skills(pool: pd.DataFrame) -> None:
    spec = FewShotSpec(num_examples=4, same_skill=False, example="{{statement}} {{answer}}")
    assert len(select_few_shot_examples(pool, spec, skill="unused", seed=SEED)) == 4


def test_selection_without_candidates_raises(pool: pd.DataFrame) -> None:
    with pytest.raises(FewShotError, match="No few-shot candidates"):
        select_few_shot_examples(pool, DIRECT_SPEC, skill="unknown", seed=SEED)


def test_steps_are_rendered_when_non_empty(pool: pd.DataFrame) -> None:
    examples = select_few_shot_examples(pool, STEPS_SPEC, skill="geometry", seed=SEED)
    assert all(example["solution"].startswith("Кроки: ") for example in examples)
    assert all(", " in example["solution"] for example in examples)


def test_empty_steps_give_empty_solution(pool: pd.DataFrame) -> None:
    examples = select_few_shot_examples(pool, STEPS_SPEC, skill="percent", seed=SEED)
    assert len(examples) == 3
    assert all(example["solution"] == "" for example in examples)


def test_require_solution_rejects_skills_without_solutions(pool: pd.DataFrame) -> None:
    with pytest.raises(FewShotError, match="with a solution"):
        select_few_shot_examples(pool, STRICT_SPEC, skill="percent", seed=SEED)


def test_examples_with_solutions_are_preferred(pool: pd.DataFrame) -> None:
    percent_ids = sorted(pool.loc[pool["skill"] == "percent", "id"])
    worked = {percent_ids[-1]: "Розв'язання з JSONL"}
    examples = select_few_shot_examples(pool, STEPS_SPEC, skill="percent", seed=SEED, worked_solutions=worked)
    assert examples[0]["solution"] == "Розв'язання з JSONL"
    assert all(example["solution"] == "" for example in examples[1:])


def test_worked_solution_takes_priority_over_steps(pool: pd.DataFrame) -> None:
    row = pool[pool["skill"] == "geometry"].iloc[0].to_dict()
    assert render_solution(row, STEPS_SPEC, {row["id"]: "Повне розв'язання"}) == "Повне розв'язання"
    assert render_solution(row, STEPS_SPEC, {}).startswith("Кроки: ")


def test_rendered_cot_example_drops_empty_solution_line(pool: pd.DataFrame) -> None:
    template = load_prompt(REPO_ROOT / "prompts", "few_shot_cot")
    example = {"statement": "Задача А", "skill_title": "Т", "answer": "5", "solution": ""}
    messages = template.render_messages(statement="Нова", skill_title="Т", examples=[example])
    assert messages[2] == {"role": "assistant", "content": "Відповідь: 5"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("[]", []), ("", []), ('["1428", "16320"]', ["1428", "16320"]), ('[" 3/4 ", ""]', ["3/4"])],
)
def test_parse_steps(raw: str, expected: list[str]) -> None:
    assert parse_steps(raw) == expected


@pytest.mark.parametrize("raw", ["not json", '{"a": 1}', "[1, 2]"])
def test_parse_steps_rejects_malformed(raw: str) -> None:
    with pytest.raises(FewShotError):
        parse_steps(raw)


def test_load_worked_solutions_accepts_train_ids(pool: pd.DataFrame, tmp_path: Path) -> None:
    first, second = pool["id"].iloc[0], pool["id"].iloc[1]
    gold = pool.set_index("id")["answer"]
    path = _write_jsonl(
        tmp_path / "worked.jsonl",
        [
            {"id": first, "solution": "  Крок 1.\nКрок 2.  ", "answer": gold[first], "source": "teacher"},
            {"id": second, "solution": "Без відповіді"},
        ],
    )
    assert load_worked_solutions(path, pool) == {first: "Крок 1.\nКрок 2.", second: "Без відповіді"}


def test_load_worked_solutions_skips_mismatched_answers(
    pool: pd.DataFrame, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    row_id = pool["id"].iloc[0]
    path = _write_jsonl(tmp_path / "worked.jsonl", [{"id": row_id, "solution": "x", "answer": "999999"}])
    with caplog.at_level(logging.INFO):
        assert load_worked_solutions(path, pool) == {}
    assert "skipped 1" in caplog.text


def test_load_worked_solutions_rejects_validation_ids(pool: pd.DataFrame, split, tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "worked.jsonl", [{"id": split.val_ids[0], "solution": "x"}])
    with pytest.raises(FewShotError, match="train split only"):
        load_worked_solutions(path, pool)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", "invalid JSON"),
        ("[1]", "JSON object"),
        ('{"solution": "x"}', "'id'"),
        ('{"id": "ID", "solution": " "}', "'solution'"),
        ('{"id": "ID", "solution": "x", "answer": 5}', "'answer'"),
        ('{"id": "ID", "solution": "Крок.\\nВідповідь: 5"}', "answer marker"),
    ],
)
def test_load_worked_solutions_rejects_malformed_records(
    pool: pd.DataFrame, tmp_path: Path, line: str, message: str
) -> None:
    path = tmp_path / "worked.jsonl"
    path.write_text(line.replace("ID", pool["id"].iloc[0]) + "\n", encoding="utf-8")
    with pytest.raises(FewShotError, match=message):
        load_worked_solutions(path, pool)


def test_load_worked_solutions_rejects_duplicates(pool: pd.DataFrame, tmp_path: Path) -> None:
    row_id = pool["id"].iloc[0]
    path = _write_jsonl(tmp_path / "worked.jsonl", [{"id": row_id, "solution": "a"}, {"id": row_id, "solution": "b"}])
    with pytest.raises(FewShotError, match="duplicate"):
        load_worked_solutions(path, pool)


def test_load_worked_solutions_missing_file(pool: pd.DataFrame, tmp_path: Path) -> None:
    with pytest.raises(FewShotError, match="not found"):
        load_worked_solutions(tmp_path / "absent.jsonl", pool)


def test_shipped_cot_prompt_errors_instead_of_falling_back(pool: pd.DataFrame) -> None:
    spec = load_prompt(REPO_ROOT / "prompts", "few_shot_cot").few_shot
    assert spec is not None
    with pytest.raises(FewShotError, match="with a solution available for skill 'percent'"):
        select_few_shot_examples(pool, spec, skill="percent", seed=SEED)
    examples = select_few_shot_examples(pool, spec, skill="geometry", seed=SEED)
    assert examples and all(example["solution"] for example in examples)
