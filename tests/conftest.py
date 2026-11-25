from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

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
