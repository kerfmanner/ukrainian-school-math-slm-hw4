from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import REPO_ROOT
from math_solver.config import ALLOWED_SUBMISSION_MODEL_IDS, ConfigError, load_config, parse_config

BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


@pytest.fixture
def raw_config() -> dict[str, Any]:
    return yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))


def test_base_config_loads() -> None:
    config = load_config(BASE_CONFIG_PATH)
    assert config.seed == 42
    assert config.split.val_ratio == pytest.approx(0.2)
    assert config.split.stratify_column == "skill"
    assert config.paths.train_csv == Path("data/raw/train.csv")
    assert config.models.default_model_id == "google/gemma-3-4b-it"
    assert set(config.models.candidates.values()) == ALLOWED_SUBMISSION_MODEL_IDS
    assert config.submission.fallback_answer == "0"
    assert config.submission.expected_rows == 3561
    assert config.prompts.directory == Path("prompts")
    assert config.prompts.default == "baseline_cot"
    assert config.extraction.last_number_fallback is True
    assert config.paths.worked_solutions_jsonl is None
    assert config.sandbox.timeout_seconds == pytest.approx(2.0)
    assert config.sandbox.allowed_modules == ("math", "fractions", "decimal")


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def _mutated(raw: dict[str, Any], section: str, key: str, value: Any) -> dict[str, Any]:
    mutated = copy.deepcopy(raw)
    mutated[section][key] = value
    return mutated


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("split", "val_ratio", 1.5, "val_ratio"),
        ("split", "val_ratio", "0.2", "must be a number"),
        ("split", "unexpected", 1, "unknown keys"),
        ("models", "candidates", {"gemma3_4b_it": "Qwen/Qwen3-8B"}, "non-allowed"),
        ("models", "default", "missing_model", "not one of"),
        ("submission", "fallback_answer", "0.0", "canonical"),
        ("submission", "expected_rows", 0, "positive"),
        ("extraction", "last_number_fallback", "yes", "must be bool"),
        ("prompts", "default", " ", "non-empty"),
        ("sandbox", "allowed_modules", ["math", "os"], "unsafe modules"),
        ("sandbox", "allowed_modules", "math", "list of strings"),
        ("paths", "worked_solutions_jsonl", 5, "must be str"),
    ],
)
def test_invalid_config_values_raise(
    raw_config: dict[str, Any], section: str, key: str, value: Any, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(_mutated(raw_config, section, key, value))


def test_missing_section_raises(raw_config: dict[str, Any]) -> None:
    del raw_config["paths"]
    with pytest.raises(ConfigError, match="missing keys"):
        parse_config(raw_config)


def test_worked_solutions_path_is_optional(raw_config: dict[str, Any]) -> None:
    del raw_config["paths"]["worked_solutions_jsonl"]
    assert parse_config(raw_config).paths.worked_solutions_jsonl is None
    raw_config["paths"]["worked_solutions_jsonl"] = "data/processed/worked.jsonl"
    assert parse_config(raw_config).paths.worked_solutions_jsonl == Path("data/processed/worked.jsonl")


def test_seed_must_be_int(raw_config: dict[str, Any]) -> None:
    raw_config["seed"] = True
    with pytest.raises(ConfigError, match="seed"):
        parse_config(raw_config)
