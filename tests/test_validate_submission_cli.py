from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from conftest import REPO_ROOT

SAMPLE_IDS = ["d455c8", "749339", "02af4f", "000123"]


@pytest.fixture
def cli_paths(tmp_path: Path) -> tuple[Path, Path]:
    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": SAMPLE_IDS, "answer": ["0"] * len(SAMPLE_IDS)}).to_csv(sample, index=False)
    config = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    config["paths"]["sample_submission_csv"] = str(sample)
    config["submission"]["expected_rows"] = len(SAMPLE_IDS)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, tmp_path / "submission.csv"


def _run(config_path: Path, submission: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_submission.py"),
            "--config",
            str(config_path),
            "--submission",
            str(submission),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_valid_submission_reports_fallback_count(cli_paths: tuple[Path, Path]) -> None:
    config_path, submission = cli_paths
    submission.write_text("id,answer\nd455c8,12\n749339,0\n02af4f,9/10\n000123,0\n", encoding="utf-8")
    result = _run(config_path, submission)
    assert result.returncode == 0, result.stderr
    assert "fallback('0')=2 (50.0%)" in result.stderr
    assert "VALID" in result.stderr


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("id,answer\nd455c8,12\n749339,0\n000123,0\n02af4f,1\n", "Id mismatch at row 2"),
        ("id,answer\nd455c8,12\n749339,0\n02af4f,1\n", "3 rows, expected 4"),
        ("id,prediction\nd455c8,12\n749339,0\n02af4f,1\n000123,1\n", "columns"),
        (",id,answer\n0,d455c8,12\n1,749339,0\n2,02af4f,1\n3,000123,1\n", "columns"),
        ("id,answer\nd455c8,\n749339,0\n02af4f,1\n000123,1\n", "1 missing answers"),
        ('id,answer\nd455c8,"17,7"\n749339,3.50\n02af4f,18/20\n000123,1\n', "3 answers not in canonical form"),
    ],
)
def test_invalid_submissions_fail(cli_paths: tuple[Path, Path], content: str, message: str) -> None:
    config_path, submission = cli_paths
    submission.write_text(content, encoding="utf-8")
    result = _run(config_path, submission)
    assert result.returncode == 1
    assert message in result.stderr
    assert "INVALID" in result.stderr


def test_expected_rows_override_is_checked(cli_paths: tuple[Path, Path]) -> None:
    config_path, submission = cli_paths
    submission.write_text("id,answer\nd455c8,1\n749339,1\n02af4f,1\n000123,1\n", encoding="utf-8")
    result = _run(config_path, submission, "--expected-rows", "3561")
    assert result.returncode == 1
    assert "expected 3561" in result.stderr


def test_missing_submission_file_fails(cli_paths: tuple[Path, Path]) -> None:
    config_path, submission = cli_paths
    result = _run(config_path, submission)
    assert result.returncode == 1
    assert "not found" in result.stderr
