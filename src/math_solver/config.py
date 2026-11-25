from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from math_solver.normalize import normalize_answer
from math_solver.structured import ConfigError, build_dataclass, load_yaml_mapping
from math_solver.tool_use import SandboxSettings

__all__ = [
    "ALLOWED_SUBMISSION_MODEL_IDS",
    "ConfigError",
    "ExtractionConfig",
    "ModelsConfig",
    "PathsConfig",
    "ProjectConfig",
    "PromptsConfig",
    "SandboxSettings",
    "SplitConfig",
    "SubmissionConfig",
    "load_config",
    "parse_config",
]

ALLOWED_SUBMISSION_MODEL_IDS: frozenset[str] = frozenset(
    {
        "google/gemma-3-4b-it",
        "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0",
    }
)


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    processed_dir: Path
    outputs_dir: Path
    train_csv: Path
    test_csv: Path
    sample_submission_csv: Path
    split_file: Path
    worked_solutions_jsonl: Path | None = None


@dataclass(frozen=True)
class SplitConfig:
    val_ratio: float
    stratify_column: str

    def __post_init__(self) -> None:
        if not 0.0 < self.val_ratio < 1.0:
            raise ConfigError(f"split.val_ratio must be in (0, 1), got {self.val_ratio}")
        if not self.stratify_column.strip():
            raise ConfigError("split.stratify_column must be a non-empty string")


@dataclass(frozen=True)
class ModelsConfig:
    default: str
    candidates: dict[str, str]

    def __post_init__(self) -> None:
        if self.default not in self.candidates:
            raise ConfigError(f"models.default '{self.default}' is not one of {sorted(self.candidates)}")
        disallowed = sorted(set(self.candidates.values()) - ALLOWED_SUBMISSION_MODEL_IDS)
        if disallowed:
            raise ConfigError(f"models.candidates contains non-allowed submission models: {disallowed}")

    @property
    def default_model_id(self) -> str:
        return self.candidates[self.default]


@dataclass(frozen=True)
class PromptsConfig:
    directory: Path
    default: str

    def __post_init__(self) -> None:
        if not self.default.strip():
            raise ConfigError("prompts.default must be a non-empty string")


@dataclass(frozen=True)
class ExtractionConfig:
    last_number_fallback: bool


@dataclass(frozen=True)
class SubmissionConfig:
    fallback_answer: str
    expected_rows: int

    def __post_init__(self) -> None:
        if normalize_answer(self.fallback_answer) != self.fallback_answer:
            raise ConfigError(f"submission.fallback_answer '{self.fallback_answer}' is not in canonical form")
        if self.expected_rows <= 0:
            raise ConfigError(f"submission.expected_rows must be positive, got {self.expected_rows}")


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    paths: PathsConfig
    split: SplitConfig
    models: ModelsConfig
    prompts: PromptsConfig
    extraction: ExtractionConfig
    sandbox: SandboxSettings
    submission: SubmissionConfig


def load_config(path: Path) -> ProjectConfig:
    return parse_config(load_yaml_mapping(path))


def parse_config(raw: Any) -> ProjectConfig:
    return build_dataclass(ProjectConfig, raw)
