from __future__ import annotations

import json
import logging
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from math_solver.data_io import require_columns
from math_solver.extract import DEFAULT_ANSWER_MARKER, marker_segments
from math_solver.normalize import normalize_answer
from math_solver.prompts import STEPS_SOURCE, FewShotSpec, substitute
from math_solver.split import ID_COLUMN, SplitResult, apply_split

EXAMPLE_FIELDS: tuple[str, ...] = ("statement", "skill_title", "answer")
SKILL_COLUMN = "skill"
STEPS_COLUMN = "steps"
ANSWER_COLUMN = "answer"
SOLUTION_FIELD = "solution"
WORKED_SOURCE = "worked"
WORKED_ID_KEY = "id"
WORKED_SOLUTION_KEY = "solution"
WORKED_ANSWER_KEY = "answer"

logger = logging.getLogger(__name__)


class FewShotError(ValueError):
    pass


def build_few_shot_pool(train_frame: pd.DataFrame, split: SplitResult) -> pd.DataFrame:
    train_part, _ = apply_split(train_frame, split)
    return train_part


def load_worked_solutions(
    path: Path,
    pool: pd.DataFrame,
    *,
    marker: str = DEFAULT_ANSWER_MARKER,
) -> dict[str, str]:
    require_columns(pool, [ID_COLUMN, ANSWER_COLUMN], "Few-shot pool")
    gold = dict(zip(pool[ID_COLUMN].astype(str), pool[ANSWER_COLUMN].astype(str)))
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise FewShotError(f"Worked solutions file not found: {path}") from error
    solutions: dict[str, str] = {}
    mismatched = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        location = f"{path}:{line_number}"
        row_id, solution, answer = _parse_worked_record(line, location)
        if row_id not in gold:
            raise FewShotError(f"{location}: id '{row_id}' is not in the few-shot pool (train split only)")
        if row_id in solutions:
            raise FewShotError(f"{location}: duplicate id '{row_id}'")
        if marker_segments(solution, marker):
            raise FewShotError(f"{location}: solution must not contain the final answer marker '{marker}'")
        if answer is not None and normalize_answer(answer) != gold[row_id]:
            mismatched += 1
            continue
        solutions[row_id] = solution
    logger.info("Loaded %d worked solutions from %s, skipped %d with mismatched answers", len(solutions), path, mismatched)
    return solutions


def parse_steps(raw: str) -> list[str]:
    if not raw.strip():
        return []
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as error:
        raise FewShotError(f"steps value is not valid JSON: {raw[:50]!r}") from error
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise FewShotError(f"steps value must be a JSON list of strings: {raw[:50]!r}")
    return [value.strip() for value in values if value.strip()]


def render_solution(row: Mapping[str, Any], spec: FewShotSpec, worked_solutions: Mapping[str, str]) -> str:
    for source in spec.solution_sources:
        if source == WORKED_SOURCE:
            worked = worked_solutions.get(str(row[ID_COLUMN]), "")
            if worked:
                return worked
        if source == STEPS_SOURCE and spec.steps_template is not None:
            steps = parse_steps(str(row[STEPS_COLUMN]))
            if steps:
                return substitute(spec.steps_template, {"steps": spec.steps_separator.join(steps)}).strip()
    return ""


def select_few_shot_examples(
    pool: pd.DataFrame,
    spec: FewShotSpec,
    *,
    skill: str,
    seed: int,
    exclude_ids: Collection[str] = (),
    worked_solutions: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    required = [ID_COLUMN, SKILL_COLUMN, *EXAMPLE_FIELDS]
    if STEPS_SOURCE in spec.solution_sources:
        required.append(STEPS_COLUMN)
    require_columns(pool, required, "Few-shot pool")
    candidates = pool[~pool[ID_COLUMN].astype(str).isin(set(exclude_ids))]
    if spec.same_skill:
        candidates = candidates[candidates[SKILL_COLUMN] == skill]
    shuffled = candidates.sort_values(ID_COLUMN, kind="stable").sample(frac=1, random_state=seed)
    examples = [_to_example(row, spec, worked_solutions or {}) for row in shuffled.to_dict("records")]
    with_solution = [example for example in examples if example[SOLUTION_FIELD]]
    without_solution = [example for example in examples if not example[SOLUTION_FIELD]]
    ranked = with_solution if spec.require_solution else with_solution + without_solution
    if not ranked:
        requirement = " with a solution" if spec.require_solution else ""
        raise FewShotError(f"No few-shot candidates{requirement} available for skill '{skill}'")
    return ranked[: spec.num_examples]


def _to_example(row: Mapping[str, Any], spec: FewShotSpec, worked_solutions: Mapping[str, str]) -> dict[str, str]:
    example = {field: str(row[field]) for field in EXAMPLE_FIELDS}
    example[SOLUTION_FIELD] = render_solution(row, spec, worked_solutions)
    return example


def _parse_worked_record(line: str, location: str) -> tuple[str, str, str | None]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise FewShotError(f"{location}: invalid JSON: {error}") from error
    if not isinstance(record, dict):
        raise FewShotError(f"{location}: each line must be a JSON object")
    row_id = record.get(WORKED_ID_KEY)
    solution = record.get(WORKED_SOLUTION_KEY)
    answer = record.get(WORKED_ANSWER_KEY)
    if not isinstance(row_id, str) or not row_id:
        raise FewShotError(f"{location}: '{WORKED_ID_KEY}' must be a non-empty string")
    if not isinstance(solution, str) or not solution.strip():
        raise FewShotError(f"{location}: '{WORKED_SOLUTION_KEY}' must be a non-empty string")
    if answer is not None and not isinstance(answer, str):
        raise FewShotError(f"{location}: '{WORKED_ANSWER_KEY}' must be a string when present")
    return row_id, solution.strip(), answer
