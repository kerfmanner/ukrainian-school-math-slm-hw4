from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from conftest import MINI_TEST_IDS, REPO_ROOT, MiniProject
from math_solver.generation import MockBackend, SamplingConfig
from math_solver.inference import (
    RunSettings,
    build_few_shot_context,
    load_problems,
    render_problem_messages,
    run_generation,
)
from math_solver.prompts import load_prompt
from math_solver.run_report import (
    ReportError,
    ScoredRun,
    build_report,
    default_report_path,
    render_report_html,
    run_labels,
    score_run_file,
    write_report,
)
from math_solver.scoring import evaluate_scored, load_val_references

MODEL_ID = "google/gemma-3-4b-it"
STATEMENT_PATTERN = re.compile(r"Задача: (.+)")
DATA_PATTERN = re.compile(r'<script type="application/json" id="report-data">(.*?)</script>', re.DOTALL)
GREEDY = SamplingConfig(max_new_tokens=64, temperature=0.0, top_p=1.0, top_k=0, num_samples=1)
SAMPLED = SamplingConfig(max_new_tokens=64, temperature=0.7, top_p=1.0, top_k=0, num_samples=3)
RUN_INFERENCE = REPO_ROOT / "scripts" / "run_inference.py"
RENDER_REPORT = REPO_ROOT / "scripts" / "render_run_report.py"

Responder = Callable[[str, int], str]


def _gold(project: MiniProject, prompt: str) -> str:
    match = STATEMENT_PATTERN.search(prompt)
    assert match is not None
    return project.gold_by_statement().get(match.group(1), "0")


def _write_run(
    project: MiniProject,
    path: Path,
    responder: Responder,
    *,
    split: str = "val",
    prompt: str = "baseline_cot",
    sampling: SamplingConfig = GREEDY,
    limit: int | None = None,
) -> Path:
    config = project.config
    template = load_prompt(config.prompts.directory, prompt)
    problems = load_problems(config, split, limit)
    few_shot = build_few_shot_context(config, template, config.seed)
    messages = {problem.id: render_problem_messages(template, problem, few_shot) for problem in problems}
    settings = RunSettings(split=split, prompt_name=prompt, sampling=sampling, batch_size=4, seed=config.seed)
    run_generation(problems, messages, MockBackend(model_id=MODEL_ID, responder=responder), settings, path)
    return path


def _scored(project: MiniProject, path: Path) -> ScoredRun:
    return score_run_file(project.config, path)


def _embedded_data(page: str) -> dict:
    match = DATA_PATTERN.search(page)
    assert match is not None
    return json.loads(match.group(1))


def test_single_val_run_rows_match_scoring(mini_project: MiniProject, tmp_path: Path) -> None:
    wrong_statement = load_problems(mini_project.config, "val")[0].statement

    def responder(prompt: str, index: int) -> str:
        answer = "999" if wrong_statement in prompt else _gold(mini_project, prompt)
        return f"Міркування.\nВідповідь: {answer}"

    run = _scored(mini_project, _write_run(mini_project, tmp_path / "gemma__baseline_cot__val.jsonl", responder))
    report = build_report(mini_project.config, [run])
    references = load_val_references(mini_project.config.paths.train_csv, mini_project.config.paths.split_file)
    expected = evaluate_scored(run.scored, references)
    assert report["has_gold"] and report["comparison"] is None
    assert report["runs"][0]["metrics"]["accuracy"] == expected.accuracy
    assert [row["id"] for row in report["problems"]] == [problem.id for problem in load_problems(mini_project.config, "val")]
    wrong = [row for row in report["problems"] if row["results"][0]["correct"] is False]
    assert len(wrong) == 1
    assert wrong[0]["statement"] == wrong_statement
    assert wrong[0]["gold"] == references[wrong[0]["id"]].answer
    assert wrong[0]["results"][0]["prediction"] == "999"
    assert "Відповідь: 999" in wrong[0]["results"][0]["samples"][0]["completion"]


def test_comparison_joins_runs_by_id(mini_project: MiniProject, tmp_path: Path) -> None:
    val_ids = [problem.id for problem in load_problems(mini_project.config, "val")]
    right = _write_run(mini_project, tmp_path / "gemma__cot__val.jsonl", lambda p, i: f"Відповідь: {_gold(mini_project, p)}")
    wrong = _write_run(mini_project, tmp_path / "mamay__cot__val.jsonl", lambda p, i: "Відповідь: 999", limit=4)
    report = build_report(mini_project.config, [_scored(mini_project, right), _scored(mini_project, wrong)])
    comparison = report["comparison"]
    assert [run["label"] for run in report["runs"]] == ["gemma", "mamay"]
    assert comparison["common_problems"] == 4
    assert comparison["answers_differ"] == 4
    assert comparison["common_accuracy"] == [1.0, 0.0]
    assert comparison["unique_correct"] == [4, 0]
    assert comparison["all_correct"] == 0 and comparison["none_correct"] == 0
    assert len(report["problems"]) == len(val_ids)
    assert sum(row["results"][1] is None for row in report["problems"]) == len(val_ids) - 4


def test_test_split_report_has_no_gold(mini_project: MiniProject, tmp_path: Path) -> None:
    path = _write_run(mini_project, tmp_path / "gemma__cot__test.jsonl", lambda p, i: "Відповідь: 5", split="test")
    report = build_report(mini_project.config, [_scored(mini_project, path)])
    assert not report["has_gold"]
    assert report["runs"][0]["metrics"] is None
    assert [row["id"] for row in report["problems"]] == sorted(MINI_TEST_IDS)
    assert all(row["gold"] is None and row["results"][0]["correct"] is None for row in report["problems"])
    assert report["problems"][0]["statement"].startswith("Тестова задача")


def test_multi_sample_results_keep_votes_and_samples(mini_project: MiniProject, tmp_path: Path) -> None:
    def responder(prompt: str, index: int) -> str:
        return "Відповідь: 999" if index == 2 else f"Відповідь: {_gold(mini_project, prompt)}"

    path = _write_run(mini_project, tmp_path / "sc.jsonl", responder, prompt="self_consistency_cot", sampling=SAMPLED)
    report = build_report(mini_project.config, [_scored(mini_project, path)])
    result = report["problems"][0]["results"][0]
    assert (result["votes"], result["num_samples"], len(result["samples"])) == (2, 3, 3)
    assert result["distribution"][1] == ["999", 1]
    assert report["runs"][0]["mean_agreement"] == pytest.approx(2 / 3)


def test_tool_use_report_includes_code_and_sandbox_output(mini_project: MiniProject, tmp_path: Path) -> None:
    def responder(prompt: str, index: int) -> str:
        return f"Код:\n```python\nanswer = 1 / 0\n```\nВідповідь: {_gold(mini_project, prompt)}"

    path = _write_run(mini_project, tmp_path / "tool.jsonl", responder, prompt="tool_use_python")
    report = build_report(mini_project.config, [_scored(mini_project, path)])
    sample = report["problems"][0]["results"][0]["samples"][0]
    assert sample["code"] == "answer = 1 / 0"
    assert sample["tool_status"] == "error"
    assert "ZeroDivisionError" in sample["tool_detail"]
    assert report["runs"][0]["use_tool"] and report["runs"][0]["tool_statuses"] == {"error": 9}


def test_html_escapes_model_text(mini_project: MiniProject, tmp_path: Path) -> None:
    hostile = "</script><script>alert(1)</script><!-- & <b>bold</b>"
    path = _write_run(mini_project, tmp_path / "run.jsonl", lambda p, i: f"{hostile}\nВідповідь: 1")
    report = build_report(mini_project.config, [_scored(mini_project, path)])
    output = tmp_path / "report.html"
    write_report(output, report)
    page = output.read_text(encoding="utf-8")
    assert page.count("<script") == 2 and page.count("</script>") == 2
    assert "<b>bold</b>" not in page
    data = _embedded_data(page)
    assert data["problems"][0]["results"][0]["samples"][0]["completion"].startswith(hostile)


def test_max_completion_chars_clips_text(mini_project: MiniProject, tmp_path: Path) -> None:
    path = _write_run(mini_project, tmp_path / "run.jsonl", lambda p, i: "а" * 500 + "\nВідповідь: 1")
    report = build_report(mini_project.config, [_scored(mini_project, path)], max_completion_chars=100)
    result = report["problems"][0]["results"][0]
    assert result["samples"][0]["completion"].startswith("а" * 100 + "\n…[")
    assert "characters omitted" in result["prompt_text"]
    assert result["prediction"] == "1"


def test_incompatible_runs_are_rejected(mini_project: MiniProject, tmp_path: Path) -> None:
    val = _scored(mini_project, _write_run(mini_project, tmp_path / "a.jsonl", lambda p, i: "Відповідь: 1"))
    test = _scored(mini_project, _write_run(mini_project, tmp_path / "b.jsonl", lambda p, i: "Відповідь: 1", split="test"))
    with pytest.raises(ReportError, match="same split"):
        build_report(mini_project.config, [val, test])
    with pytest.raises(ReportError, match="unique"):
        build_report(mini_project.config, [val, val])
    with pytest.raises(ReportError, match="max_completion_chars"):
        build_report(mini_project.config, [val], max_completion_chars=0)


def test_template_placeholders_are_required() -> None:
    with pytest.raises(ReportError, match="exactly once"):
        render_report_html({"title": "x"}, "<html>__REPORT_TITLE__</html>")


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["gemma__cot__val__smoke20"], ["gemma__cot__val__smoke20"]),
        (["gemma__cot__val__smoke20", "mamay__cot__val__smoke20"], ["gemma", "mamay"]),
        (["gemma__cot__val", "gemma__tool__val"], ["cot", "tool"]),
        (["a__x", "x__a"], ["a__x", "x__a"]),
    ],
)
def test_run_labels(names: list[str], expected: list[str]) -> None:
    assert run_labels(names) == expected


def test_default_report_path(tmp_path: Path) -> None:
    assert default_report_path(tmp_path, ["a", "b"]) == tmp_path / "a__vs__b__report.html"


def _cli(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, check=False)


def test_render_cli_single_and_comparison(mini_project: MiniProject) -> None:
    config = str(mini_project.config_path)
    runs = []
    for tag in ("first", "second"):
        run_file = mini_project.root / "runs" / f"mock__baseline_cot__val__{tag}.jsonl"
        generated = _cli(RUN_INFERENCE, "--config", config, "--backend", "mock", "--split", "val", "--output", str(run_file))
        assert generated.returncode == 0, generated.stderr
        runs.append(run_file)
    single = _cli(RENDER_REPORT, "--config", config, "--run", str(runs[0]))
    assert single.returncode == 0, single.stderr
    single_page = mini_project.root / "outputs" / "report_views" / "mock__baseline_cot__val__first__report.html"
    assert _embedded_data(single_page.read_text(encoding="utf-8"))["title"] == runs[0].stem
    output = mini_project.root / "compare.html"
    compared = _cli(RENDER_REPORT, "--config", config, "--run", str(runs[0]), "--run", str(runs[1]), "--output", str(output))
    assert compared.returncode == 0, compared.stderr
    data = _embedded_data(output.read_text(encoding="utf-8"))
    assert data["title"] == "first vs second"
    assert data["comparison"]["answers_differ"] == 0
    missing = _cli(RENDER_REPORT, "--config", config, "--run", str(mini_project.root / "missing.jsonl"))
    assert missing.returncode == 1 and "Run file not found" in missing.stderr
