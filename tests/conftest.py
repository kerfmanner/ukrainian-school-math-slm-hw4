from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
import yaml

from math_solver.config import ProjectConfig, load_config
from math_solver.split import make_stratified_split, save_split

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS = ("addition", "division", "fractions", "geometry", "percent")
ROWS_PER_SKILL = 10


@pytest.fixture
def train_frame() -> pd.DataFrame:
    rows = [
        {
            "id": f"{skill}-{index}",
            "skill": skill,
            "skill_title": skill.title(),
            "statement": f"Задача {index}",
            "answer": str(index),
        }
        for skill in SKILLS
        for index in range(ROWS_PER_SKILL)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def sample_submission() -> pd.DataFrame:
    return pd.DataFrame({"id": ["101", "7", "55", "3"], "answer": ["0", "0", "0", "0"]})


MINI_SKILLS = ("A-ADD", "B-MUL", "C-FRAC")
MINI_ROWS_PER_SKILL = 6
MINI_TEST_IDS = ("t3", "t1", "t4", "t0", "t2")


@dataclass(frozen=True)
class MiniProject:
    root: Path
    config_path: Path
    train: pd.DataFrame

    @property
    def config(self) -> ProjectConfig:
        return load_config(self.config_path)

    def gold_by_statement(self) -> dict[str, str]:
        return dict(zip(self.train["statement"], self.train["answer"]))


@pytest.fixture
def mini_project(tmp_path: Path) -> MiniProject:
    train = pd.DataFrame(
        [
            {
                "id": f"{skill.lower()}-{index}",
                "skill": skill,
                "skill_title": f"Тема {skill}",
                "statement": f"Обчисли {skill} номер {index}.",
                "answer": str(10 * index + len(skill)),
                "steps": "[]",
                "hints": "[]",
                "typical_errors": "[]",
            }
            for skill in MINI_SKILLS
            for index in range(MINI_ROWS_PER_SKILL)
        ]
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    train.to_csv(raw / "train.csv", index=False)
    test_ids = sorted(MINI_TEST_IDS)
    pd.DataFrame(
        {
            "id": test_ids,
            "skill": [MINI_SKILLS[index % len(MINI_SKILLS)] for index in range(len(test_ids))],
            "skill_title": ["Тема"] * len(test_ids),
            "statement": [f"Тестова задача {row_id}." for row_id in test_ids],
        }
    ).to_csv(raw / "test.csv", index=False)
    pd.DataFrame({"id": list(MINI_TEST_IDS), "answer": ["0"] * len(MINI_TEST_IDS)}).to_csv(
        raw / "sample_submission.csv", index=False
    )
    split_file = tmp_path / "processed" / "split.json"
    save_split(make_stratified_split(train, val_ratio=0.5, seed=42, stratify_column="skill"), split_file)
    config = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    config["paths"].update(
        raw_dir=str(raw),
        processed_dir=str(split_file.parent),
        outputs_dir=str(tmp_path / "outputs"),
        train_csv=str(raw / "train.csv"),
        test_csv=str(raw / "test.csv"),
        sample_submission_csv=str(raw / "sample_submission.csv"),
        split_file=str(split_file),
    )
    config["prompts"]["directory"] = str(REPO_ROOT / "prompts")
    config["submission"]["expected_rows"] = len(MINI_TEST_IDS)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return MiniProject(root=tmp_path, config_path=config_path, train=train)
