from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from conftest import MINI_TEST_IDS, REPO_ROOT, MiniProject

RUN_INFERENCE = REPO_ROOT / "scripts" / "run_inference.py"
SCORE_RUN = REPO_ROOT / "scripts" / "score_run.py"


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, check=False)


def _infer(project: MiniProject, output: Path, *args: str) -> subprocess.CompletedProcess[str]:
    base = ("--config", str(project.config_path), "--backend", "mock", "--output", str(output))
    return _run(RUN_INFERENCE, *base, *args)


def test_val_limit_resume_and_scoring(mini_project: MiniProject) -> None:
    run_file = mini_project.root / "runs" / "mock__baseline_cot__val.jsonl"
    first = _infer(mini_project, run_file, "--split", "val", "--limit", "3", "--batch-size", "2")
    assert first.returncode == 0, first.stderr
    assert len(run_file.read_text(encoding="utf-8").splitlines()) == 3
    second = _infer(mini_project, run_file, "--split", "val", "--batch-size", "2")
    assert second.returncode == 0, second.stderr
    assert "resumed=3" in second.stderr and "generated=6" in second.stderr
    assert "health: problems=9" in second.stderr
    report_dir = mini_project.root / "reports"
    config = str(mini_project.config_path)
    scored = _run(SCORE_RUN, "--config", config, "--run", str(run_file), "--report-dir", str(report_dir))
    assert scored.returncode == 0, scored.stderr
    assert all(text in scored.stderr for text in ("accuracy=", "per-skill accuracy", "per-method accuracy"))
    metrics = json.loads((report_dir / "mock__baseline_cot__val__metrics.json").read_text(encoding="utf-8"))
    assert metrics["evaluated"] == 9 and metrics["val_rows"] == 9
    errors = (report_dir / "mock__baseline_cot__val__errors.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(errors) == metrics["evaluated"] - metrics["correct"]


def test_changed_settings_refuse_to_resume(mini_project: MiniProject) -> None:
    run_file = mini_project.root / "run.jsonl"
    assert _infer(mini_project, run_file, "--split", "val", "--limit", "2").returncode == 0
    result = _infer(mini_project, run_file, "--split", "val", "--max-new-tokens", "16")
    assert result.returncode == 1
    assert "different settings" in result.stderr


def test_multi_sample_cli(mini_project: MiniProject) -> None:
    run_file = mini_project.root / "run.jsonl"
    sampling = ("--num-samples", "3", "--temperature", "0.8")
    result = _infer(mini_project, run_file, "--split", "val", "--limit", "2", *sampling)
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in run_file.read_text(encoding="utf-8").splitlines()]
    assert all(len(record["completions"]) == 3 for record in records)
    assert all(record["generation_params"]["temperature"] == 0.8 for record in records)


def test_test_split_builds_validated_submission(mini_project: MiniProject) -> None:
    run_file = mini_project.root / "run.jsonl"
    assert _infer(mini_project, run_file, "--split", "test", "--prompt", "tool_use_python").returncode == 0
    submission = mini_project.root / "submission.csv"
    args = ("--config", str(mini_project.config_path), "--run", str(run_file), "--submission", str(submission))
    refused = _run(SCORE_RUN, *args)
    assert refused.returncode == 1 and "mock backend" in refused.stderr
    result = _run(SCORE_RUN, *args, "--allow-mock")
    assert result.returncode == 0, result.stderr
    assert "VALID" in result.stderr
    frame = pd.read_csv(submission, dtype=str)
    assert list(frame.columns) == ["id", "answer"]
    assert frame["id"].tolist() == list(MINI_TEST_IDS)


def test_partial_test_run_requires_flag(mini_project: MiniProject) -> None:
    run_file = mini_project.root / "run.jsonl"
    assert _infer(mini_project, run_file, "--split", "test", "--limit", "2").returncode == 0
    args = ("--config", str(mini_project.config_path), "--run", str(run_file), "--allow-mock")
    refused = _run(SCORE_RUN, *args, "--submission", str(mini_project.root / "a.csv"))
    assert refused.returncode == 1 and "missing 3 test ids" in refused.stderr
    allowed = _run(SCORE_RUN, *args, "--submission", str(mini_project.root / "b.csv"), "--allow-partial")
    assert allowed.returncode == 0, allowed.stderr
    assert "fallback('0')=3" in allowed.stderr


def test_gpu_backend_on_cpu_fails_cleanly(mini_project: MiniProject) -> None:
    config = str(mini_project.config_path)
    result = _run(RUN_INFERENCE, "--config", config, "--split", "val", "--limit", "1", "--backend", "hf")
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "gpu extra" in result.stderr or "No CUDA GPU" in result.stderr
