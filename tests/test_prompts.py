from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import REPO_ROOT
from math_solver.config import load_config
from math_solver.extract import extract_answer
from math_solver.prompts import (
    PromptError,
    available_prompts,
    load_prompt,
    placeholders,
    substitute,
)
from math_solver.structured import ConfigError

PROMPTS_DIR = REPO_ROOT / "prompts"
EXPECTED_PROMPTS = {
    "baseline_cot",
    "few_shot_cot",
    "few_shot_direct",
    "plan_then_solve",
    "self_consistency_cot",
    "tool_use_python",
}
EXAMPLES = [
    {"statement": "Обчисли: 207 : 10", "skill_title": "Множення й ділення", "answer": "207/10", "solution": "207 : 10 = 207/10"},
    {"statement": "Обчисли: 2,983 × 10", "skill_title": "Множення й ділення", "answer": "29.83", "solution": ""},
]


@pytest.fixture
def minimal_prompt() -> dict[str, Any]:
    return {
        "name": "tiny",
        "strategy": "chain_of_thought",
        "description": "test",
        "answer_marker": "Відповідь:",
        "system": "Ти вчитель.",
        "user": "Задача: {{statement}}\nВідповідь: <число>",
    }


def _write_prompt(directory: Path, payload: dict[str, Any]) -> None:
    (directory / f"{payload['name']}.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


def test_expected_prompt_files_exist() -> None:
    assert EXPECTED_PROMPTS <= set(available_prompts(PROMPTS_DIR))


def test_default_prompt_from_config_loads() -> None:
    config = load_config(REPO_ROOT / "configs" / "base.yaml")
    template = load_prompt(REPO_ROOT / config.prompts.directory, config.prompts.default)
    assert template.strategy == "chain_of_thought"


@pytest.mark.parametrize("name", sorted(EXPECTED_PROMPTS))
def test_every_prompt_renders_without_leftover_placeholders(name: str) -> None:
    template = load_prompt(PROMPTS_DIR, name)
    examples = EXAMPLES if template.few_shot is not None else ()
    messages = template.render_messages(statement="Обчисли: 2 + 3", skill_title="Додавання", examples=examples)
    assert [message["role"] for message in messages] == ["system", "user"]
    user = messages[-1]["content"]
    assert "Обчисли: 2 + 3" in user
    assert "{{" not in user and "}}" not in user
    assert template.answer_marker in user
    assert all(reference.startswith("https://") for reference in template.references)


def test_baseline_prompt_requests_parseable_marker() -> None:
    template = load_prompt(PROMPTS_DIR, "baseline_cot")
    user = template.render_messages(statement="s", skill_title="t")[-1]["content"]
    assert user.splitlines()[-1] == "В останньому рядку напиши тільки: Відповідь: <число>"
    assert extract_answer("Відповідь: 5", marker=template.answer_marker).answer == "5"


def test_direct_few_shot_prompt_renders_problem_answer_pairs_in_order() -> None:
    template = load_prompt(PROMPTS_DIR, "few_shot_direct")
    assert template.few_shot is not None and not template.few_shot.solution_sources
    user = template.render_messages(statement="Нова", skill_title="Тема", examples=EXAMPLES)[-1]["content"]
    first = user.index("Задача: Обчисли: 207 : 10\nВідповідь: 207/10")
    second = user.index("Задача: Обчисли: 2,983 × 10\nВідповідь: 29.83")
    assert first < second < user.index("Задача: Нова")
    assert "207 : 10 = 207/10" not in user


def test_cot_few_shot_prompt_renders_solutions() -> None:
    template = load_prompt(PROMPTS_DIR, "few_shot_cot")
    assert template.few_shot is not None
    assert template.few_shot.solution_sources == ("worked", "steps")
    assert template.few_shot.require_solution is True
    user = template.render_messages(statement="Нова", skill_title="Тема", examples=EXAMPLES[:1])[-1]["content"]
    assert "Задача: Обчисли: 207 : 10\nРозв'язання:\n207 : 10 = 207/10\nВідповідь: 207/10" in user


def test_few_shot_prompt_requires_examples() -> None:
    template = load_prompt(PROMPTS_DIR, "few_shot_cot")
    with pytest.raises(PromptError, match="at least one"):
        template.render_messages(statement="s", skill_title="t")


def test_zero_shot_prompt_rejects_examples() -> None:
    template = load_prompt(PROMPTS_DIR, "baseline_cot")
    with pytest.raises(PromptError, match="does not accept"):
        template.render_messages(statement="s", skill_title="t", examples=EXAMPLES)


def test_tool_use_prompt_renders_result_message() -> None:
    template = load_prompt(PROMPTS_DIR, "tool_use_python")
    assert template.tool_use is not None
    message = template.render_tool_result("17748")
    assert message.startswith("Результат виконання коду: 17748")
    assert template.answer_marker in message


def test_tool_result_requires_tool_section() -> None:
    with pytest.raises(PromptError, match="tool_use"):
        load_prompt(PROMPTS_DIR, "baseline_cot").render_tool_result("1")


def test_self_consistency_prompt_has_sampling_spec() -> None:
    spec = load_prompt(PROMPTS_DIR, "self_consistency_cot").self_consistency
    assert spec is not None
    assert spec.num_samples >= 2
    assert spec.aggregation == "majority_vote"


def test_substitute_does_not_touch_latex_braces() -> None:
    assert substitute(r"$\frac{1}{2}$ {{statement}}", {"statement": "x"}) == r"$\frac{1}{2}$ x"


def test_substitute_does_not_recurse_into_values() -> None:
    assert substitute("{{statement}}", {"statement": "{{answer}}"}) == "{{answer}}"


def test_substitute_reports_missing_values() -> None:
    with pytest.raises(PromptError, match="skill_title"):
        substitute("{{statement}} {{skill_title}}", {"statement": "x"})


def test_substitute_drops_lines_with_only_empty_placeholders() -> None:
    text = "А\n{{solution}}\n  {{solution}}  \nБ {{solution}}\nВ"
    assert substitute(text, {"solution": ""}) == "А\nБ \nВ"


def test_placeholders_tolerate_spaces() -> None:
    assert placeholders("{{ statement }} {{answer}}") == {"statement", "answer"}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"user": "Задача без змінних. Відповідь: <число>"}, "statement"),
        ({"user": "{{statement}} {{unknown}} Відповідь:"}, "unknown placeholders"),
        ({"user": "{{statement}} без маркера"}, "answer marker"),
        ({"system": "{{statement}}"}, "system"),
        ({"user": "{{statement}} {{examples}} Відповідь:"}, "few_shot"),
        ({"extra": 1}, "unknown keys"),
        ({"self_consistency": {"num_samples": 1, "temperature": 0.7, "aggregation": "majority_vote"}}, "num_samples"),
        ({"tool_use": {"language": "python", "result_message": "без результату"}}, "result"),
    ],
)
def test_invalid_prompt_files_raise(
    minimal_prompt: dict[str, Any], tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    _write_prompt(tmp_path, {**minimal_prompt, **change})
    with pytest.raises(ConfigError, match=message):
        load_prompt(tmp_path, "tiny")


@pytest.mark.parametrize(
    ("few_shot", "message"),
    [
        ({"example": "{{statement}} {{solution}} {{answer}}"}, "exactly when"),
        ({"example": "{{statement}} {{answer}}", "solution_sources": ["worked"]}, "exactly when"),
        ({"example": "{{solution}}", "solution_sources": ["teacher"]}, "unknown values"),
        ({"example": "{{solution}}", "solution_sources": ["worked", "worked"]}, "duplicates"),
        ({"example": "{{solution}}", "solution_sources": ["steps"]}, "steps_template is required"),
        ({"example": "{{solution}}", "solution_sources": ["steps"], "steps_template": "без кроків"}, "{{steps}}"),
        ({"example": "{{answer}}", "require_solution": True}, "require_solution"),
    ],
)
def test_invalid_few_shot_specs_raise(
    minimal_prompt: dict[str, Any], tmp_path: Path, few_shot: dict[str, Any], message: str
) -> None:
    payload = {
        **minimal_prompt,
        "user": "{{statement}} {{examples}} Відповідь:",
        "few_shot": {"num_examples": 2, "same_skill": True, **few_shot},
    }
    _write_prompt(tmp_path, payload)
    with pytest.raises(ConfigError, match=message):
        load_prompt(tmp_path, "tiny")


def test_few_shot_section_without_placeholder_raises(minimal_prompt: dict[str, Any], tmp_path: Path) -> None:
    few_shot = {"num_examples": 2, "same_skill": True, "example": "{{statement}} {{answer}}"}
    _write_prompt(tmp_path, {**minimal_prompt, "few_shot": few_shot})
    with pytest.raises(ConfigError, match="used together"):
        load_prompt(tmp_path, "tiny")


def test_prompt_name_must_match_file(minimal_prompt: dict[str, Any], tmp_path: Path) -> None:
    _write_prompt(tmp_path, minimal_prompt)
    (tmp_path / "tiny.yaml").rename(tmp_path / "other.yaml")
    with pytest.raises(ConfigError, match="declares name"):
        load_prompt(tmp_path, "other")


def test_missing_prompt_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_prompt(tmp_path, "absent")
