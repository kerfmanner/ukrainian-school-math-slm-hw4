from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from conftest import REPO_ROOT, ROWS_PER_SKILL, SKILLS
from math_solver.split import SplitError, apply_split, load_split, make_stratified_split, save_split

VAL_RATIO = 0.2
SEED = 42


def _split(frame: pd.DataFrame, seed: int = SEED):
    return make_stratified_split(frame, val_ratio=VAL_RATIO, seed=seed, stratify_column="skill")


def test_split_is_deterministic_for_fixed_seed(train_frame: pd.DataFrame) -> None:
    assert _split(train_frame) == _split(train_frame)


def test_split_changes_with_seed(train_frame: pd.DataFrame) -> None:
    assert _split(train_frame, seed=1).val_ids != _split(train_frame, seed=2).val_ids


def test_split_is_disjoint_and_complete(train_frame: pd.DataFrame) -> None:
    split = _split(train_frame)
    assert not set(split.train_ids) & set(split.val_ids)
    assert set(split.train_ids) | set(split.val_ids) == set(train_frame["id"])
    assert split.source_rows == len(train_frame)


def test_split_is_stratified_by_skill(train_frame: pd.DataFrame) -> None:
    split = _split(train_frame)
    val_skills = train_frame.set_index("id").loc[list(split.val_ids), "skill"].value_counts()
    expected_per_skill = int(ROWS_PER_SKILL * VAL_RATIO)
    assert val_skills.to_dict() == {skill: expected_per_skill for skill in SKILLS}


def test_split_assigns_singleton_skills_to_train(train_frame: pd.DataFrame) -> None:
    singleton = pd.DataFrame([{"id": "lonely-0", "skill": "lonely", "answer": "1"}])
    frame = pd.concat([train_frame, singleton], ignore_index=True)
    split = _split(frame)
    assert "lonely-0" in split.train_ids


def test_split_rejects_duplicate_ids(train_frame: pd.DataFrame) -> None:
    frame = pd.concat([train_frame, train_frame.iloc[:1]], ignore_index=True)
    with pytest.raises(SplitError, match="duplicate"):
        _split(frame)


def test_split_rejects_missing_skill(train_frame: pd.DataFrame) -> None:
    frame = train_frame.copy()
    frame.loc[0, "skill"] = ""
    with pytest.raises(SplitError, match="missing values"):
        _split(frame)


def test_split_round_trip_and_apply(train_frame: pd.DataFrame, tmp_path: Path) -> None:
    split = _split(train_frame)
    path = tmp_path / "split.json"
    save_split(split, path)
    loaded = load_split(path)
    assert loaded == split
    train, val = apply_split(train_frame, loaded)
    assert len(train) + len(val) == len(train_frame)
    assert val["id"].tolist() == list(split.val_ids)


def test_apply_split_rejects_changed_source(train_frame: pd.DataFrame) -> None:
    split = _split(train_frame)
    changed = train_frame.copy()
    changed.loc[0, "skill"] = "geometry"
    with pytest.raises(SplitError, match="does not match"):
        apply_split(changed, split)


def _write_cli_config(tmp_path: Path, train_csv: Path, split_file: Path) -> Path:
    config = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    config["paths"]["train_csv"] = str(train_csv)
    config["paths"]["split_file"] = str(split_file)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "make_split.py"), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_make_split_cli_creates_and_protects_split(train_frame: pd.DataFrame, tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    split_file = tmp_path / "processed" / "split.json"
    train_frame.to_csv(train_csv, index=False)
    config_path = _write_cli_config(tmp_path, train_csv, split_file)

    created = _run_cli("--config", str(config_path))
    assert created.returncode == 0, created.stderr
    assert load_split(split_file).seed == SEED

    repeated = _run_cli("--config", str(config_path))
    assert repeated.returncode == 0, repeated.stderr
    assert "identical" in repeated.stderr

    conflicting = _run_cli("--config", str(config_path), "--seed", "7")
    assert conflicting.returncode == 1
    assert "--overwrite" in conflicting.stderr

    overwritten = _run_cli("--config", str(config_path), "--seed", "7", "--overwrite")
    assert overwritten.returncode == 0, overwritten.stderr
    assert load_split(split_file).seed == 7
