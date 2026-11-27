from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from math_solver.structured import ConfigError, build_dataclass, load_yaml_mapping

PROMPT_FILE_SUFFIX = ".yaml"
PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
USER_PLACEHOLDERS = frozenset({"statement", "skill_title", "examples"})
EXAMPLE_PLACEHOLDERS = frozenset({"statement", "skill_title", "answer", "solution"})
STEPS_PLACEHOLDERS = frozenset({"steps"})
SOLUTION_SOURCES = frozenset({"worked", "steps"})
SOLUTION_PLACEHOLDER = "solution"
STEPS_SOURCE = "steps"
TOOL_RESULT_PLACEHOLDER = "result"
EXAMPLES_PLACEHOLDER = "examples"
INLINE_LAYOUT = "inline"
CHAT_LAYOUT = "chat"
FEW_SHOT_LAYOUTS = frozenset({INLINE_LAYOUT, CHAT_LAYOUT})
SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"

ChatMessage = dict[str, str]


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class FewShotSpec:
    num_examples: int
    same_skill: bool
    example: str
    layout: str = CHAT_LAYOUT
    separator: str = "\n\n"
    solution_sources: tuple[str, ...] = ()
    steps_template: str | None = None
    steps_separator: str = ", "
    require_solution: bool = False

    def __post_init__(self) -> None:
        if self.num_examples < 1:
            raise ConfigError(f"few_shot.num_examples must be >= 1, got {self.num_examples}")
        if self.layout not in FEW_SHOT_LAYOUTS:
            raise ConfigError(f"few_shot.layout must be one of {sorted(FEW_SHOT_LAYOUTS)}, got '{self.layout}'")
        example_placeholders = _require_placeholders_within(self.example, EXAMPLE_PLACEHOLDERS, "few_shot.example")
        self._validate_solution_sources(SOLUTION_PLACEHOLDER in example_placeholders)
        if self.steps_template is not None:
            found = _require_placeholders_within(self.steps_template, STEPS_PLACEHOLDERS, "few_shot.steps_template")
            if STEPS_SOURCE not in found:
                raise ConfigError("few_shot.steps_template must contain the {{steps}} placeholder")
        if STEPS_SOURCE in self.solution_sources and self.steps_template is None:
            raise ConfigError("few_shot.steps_template is required when 'steps' is a solution source")

    def _validate_solution_sources(self, uses_solution: bool) -> None:
        unknown = sorted(set(self.solution_sources) - SOLUTION_SOURCES)
        if unknown:
            raise ConfigError(f"few_shot.solution_sources has unknown values {unknown}; allowed: {sorted(SOLUTION_SOURCES)}")
        if len(set(self.solution_sources)) != len(self.solution_sources):
            raise ConfigError("few_shot.solution_sources must not contain duplicates")
        if bool(self.solution_sources) != uses_solution:
            raise ConfigError("few_shot.example must use {{solution}} exactly when solution_sources is non-empty")
        if self.require_solution and not self.solution_sources:
            raise ConfigError("few_shot.require_solution needs at least one solution source")


@dataclass(frozen=True)
class SelfConsistencySpec:
    num_samples: int
    temperature: float
    aggregation: str

    def __post_init__(self) -> None:
        if self.num_samples < 2:
            raise ConfigError(f"self_consistency.num_samples must be >= 2, got {self.num_samples}")
        if self.temperature <= 0:
            raise ConfigError(f"self_consistency.temperature must be positive, got {self.temperature}")


@dataclass(frozen=True)
class ToolUseSpec:
    language: str
    result_message: str

    def __post_init__(self) -> None:
        if TOOL_RESULT_PLACEHOLDER not in placeholders(self.result_message):
            raise ConfigError("tool_use.result_message must contain the {{result}} placeholder")


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    strategy: str
    description: str
    answer_marker: str
    system: str
    user: str
    references: tuple[str, ...] = ()
    few_shot: FewShotSpec | None = None
    self_consistency: SelfConsistencySpec | None = None
    tool_use: ToolUseSpec | None = None
    max_new_tokens: int | None = None

    def __post_init__(self) -> None:
        if placeholders(self.system):
            raise ConfigError(f"Prompt '{self.name}': system text must not contain placeholders")
        user_placeholders = _require_placeholders_within(self.user, USER_PLACEHOLDERS, f"{self.name}.user")
        if "statement" not in user_placeholders:
            raise ConfigError(f"Prompt '{self.name}': user text must contain {{{{statement}}}}")
        if (EXAMPLES_PLACEHOLDER in user_placeholders) != self._uses_inline_examples:
            raise ConfigError(
                f"Prompt '{self.name}': {{{{examples}}}} must be used exactly when few_shot.layout is '{INLINE_LAYOUT}'"
            )
        if self.answer_marker not in self.user:
            raise ConfigError(f"Prompt '{self.name}': user text must request the answer marker '{self.answer_marker}'")
        if self.few_shot is not None and self.few_shot.layout == CHAT_LAYOUT:
            self._validate_chat_example(self.few_shot)
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ConfigError(f"Prompt '{self.name}': max_new_tokens must be >= 1, got {self.max_new_tokens}")

    @property
    def _uses_inline_examples(self) -> bool:
        return self.few_shot is not None and self.few_shot.layout == INLINE_LAYOUT

    def _validate_chat_example(self, spec: FewShotSpec) -> None:
        if "answer" not in placeholders(spec.example):
            raise ConfigError(f"Prompt '{self.name}': chat few_shot.example must contain {{{{answer}}}}")
        if self.answer_marker not in spec.example:
            raise ConfigError(
                f"Prompt '{self.name}': chat few_shot.example must end the reply with '{self.answer_marker}'"
            )

    def render_messages(
        self,
        *,
        statement: str,
        skill_title: str,
        examples: Sequence[Mapping[str, str]] = (),
    ) -> list[ChatMessage]:
        if self.few_shot is None and examples:
            raise PromptError(f"Prompt '{self.name}' does not accept few-shot examples")
        if self.few_shot is not None and not examples:
            raise PromptError(f"Prompt '{self.name}' requires at least one few-shot example")
        values = {"statement": statement, "skill_title": skill_title}
        messages: list[ChatMessage] = []
        if self.system.strip():
            messages.append({"role": SYSTEM_ROLE, "content": self.system.strip()})
        if self._uses_inline_examples:
            values[EXAMPLES_PLACEHOLDER] = self._render_inline_examples(examples)
        elif self.few_shot is not None:
            messages.extend(self._render_example_turns(examples))
        messages.append({"role": USER_ROLE, "content": substitute(self.user, values).strip()})
        return messages

    def render_tool_result(self, result: str) -> str:
        if self.tool_use is None:
            raise PromptError(f"Prompt '{self.name}' has no tool_use section")
        return substitute(self.tool_use.result_message, {TOOL_RESULT_PLACEHOLDER: result}).strip()

    def _render_inline_examples(self, examples: Sequence[Mapping[str, str]]) -> str:
        if self.few_shot is None:
            raise PromptError(f"Prompt '{self.name}' has no few_shot section")
        rendered = [substitute(self.few_shot.example, example).strip() for example in examples]
        return self.few_shot.separator.join(rendered)

    def _render_example_turns(self, examples: Sequence[Mapping[str, str]]) -> list[ChatMessage]:
        if self.few_shot is None:
            raise PromptError(f"Prompt '{self.name}' has no few_shot section")
        turns: list[ChatMessage] = []
        for example in examples:
            question = {"statement": example["statement"], "skill_title": example["skill_title"]}
            turns.append({"role": USER_ROLE, "content": substitute(self.user, question).strip()})
            turns.append({"role": ASSISTANT_ROLE, "content": substitute(self.few_shot.example, example).strip()})
        return turns


def placeholders(text: str) -> set[str]:
    return set(PLACEHOLDER_PATTERN.findall(text))


def substitute(text: str, values: Mapping[str, str]) -> str:
    missing = sorted(placeholders(text) - set(values))
    if missing:
        raise PromptError(f"Missing values for placeholders: {missing}")
    rendered = [
        PLACEHOLDER_PATTERN.sub(lambda match: values[match.group(1)], line)
        for line in text.split("\n")
        if not _is_empty_placeholder_line(line, values)
    ]
    return "\n".join(rendered)


def _is_empty_placeholder_line(line: str, values: Mapping[str, str]) -> bool:
    names = PLACEHOLDER_PATTERN.findall(line)
    if not names or PLACEHOLDER_PATTERN.sub("", line).strip():
        return False
    return all(not values[name] for name in names)


def load_prompt(directory: Path, name: str) -> PromptTemplate:
    path = directory / f"{name}{PROMPT_FILE_SUFFIX}"
    template = build_dataclass(PromptTemplate, load_yaml_mapping(path))
    if template.name != name:
        raise ConfigError(f"Prompt file {path} declares name '{template.name}', expected '{name}'")
    return template


def available_prompts(directory: Path) -> list[str]:
    return sorted(path.stem for path in directory.glob(f"*{PROMPT_FILE_SUFFIX}"))


def _require_placeholders_within(text: str, allowed: frozenset[str], location: str) -> set[str]:
    found = placeholders(text)
    unknown = sorted(found - allowed)
    if unknown:
        raise ConfigError(f"'{location}' uses unknown placeholders {unknown}; allowed: {sorted(allowed)}")
    return found
