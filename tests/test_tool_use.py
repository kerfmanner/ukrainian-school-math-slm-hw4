from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from math_solver.structured import ConfigError
from math_solver.tool_use import (
    ExecutionStatus,
    SandboxSettings,
    extract_python_code,
    find_code_violation,
    run_python,
    solve_batch_with_tool,
    solve_with_tool,
)

SETTINGS = SandboxSettings(
    timeout_seconds=2.0,
    memory_limit_mb=512,
    max_code_chars=4000,
    max_output_chars=2000,
    allowed_modules=("math", "fractions", "decimal"),
)
FAST_TIMEOUT = replace(SETTINGS, timeout_seconds=0.5)


def test_extract_python_code_takes_last_python_block() -> None:
    completion = "Спершу:\n```python\nanswer = 1\n```\nВиправлення:\n```python\nanswer = 2\n```\nВідповідь: 2"
    assert extract_python_code(completion) == "answer = 2"


def test_extract_python_code_prefers_labelled_block() -> None:
    completion = "```python\nanswer = 1\n```\n```\nnot python\n```\n```text\nignored\n```"
    assert extract_python_code(completion) == "answer = 1"


@pytest.mark.parametrize("language", ["py", "Python", "python3"])
def test_extract_python_code_accepts_language_aliases(language: str) -> None:
    assert extract_python_code(f"```{language}\r\nanswer = 3\r\n```") == "answer = 3"


def test_extract_python_code_falls_back_to_unlabelled_block() -> None:
    assert extract_python_code("```\nanswer = 4\n```") == "answer = 4"


@pytest.mark.parametrize("completion", ["", "Без коду. Відповідь: 5", "```python\n\n```", "```text\nx = 1\n```"])
def test_extract_python_code_returns_none_without_code(completion: str) -> None:
    assert extract_python_code(completion) is None


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("answer = 204 * 87", "17748"),
        ("from fractions import Fraction\nanswer = Fraction(207, 10)", "207/10"),
        ("from fractions import Fraction\nanswer = Fraction(138, 10)", "69/5"),
        ("from fractions import Fraction\nanswer = Fraction(10, 5)", "2"),
        ("import math\nanswer = math.gcd(84, 126)", "42"),
        ("from math import lcm\nanswer = lcm(4, 6)", "12"),
        ("from decimal import Decimal\nanswer = Decimal('2.983') * 10", "29.83"),
        ("answer = 0.1 + 0.2", "0.3"),
        ("answer = 1.012 / 1000", "0.001012"),
        ("answer = 988.3 * 1000", "988300"),
        ("answer = 3.50", "3.5"),
        ("print(17748)", "17748"),
        ("print('проміжне 5')\nprint('Відповідь: 20,7')", "20.7"),
        ("print(1)\nanswer = 2", "2"),
        ("total = sum([12, 15, 18])\nanswer = total // 3", "15"),
        ("def area(a, b):\n    return a * b\nanswer = area(12, 5)", "60"),
        ("# коментар українською\nanswer = 7", "7"),
    ],
)
def test_run_python_returns_canonical_answer(code: str, expected: str) -> None:
    result = run_python(code, SETTINGS)
    assert result.status is ExecutionStatus.OK, result.error
    assert result.answer == expected


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("import os\nanswer = 1", "import of 'os'"),
        ("import subprocess", "import of 'subprocess'"),
        ("from os import path", "import from 'os'"),
        ("from math import *", "wildcard"),
        ("import math.sys", "import of 'math.sys'"),
        ("answer = open('/etc/passwd').read()", "name 'open'"),
        ("answer = eval('1+1')", "name 'eval'"),
        ("exec('answer = 1')", "name 'exec'"),
        ("answer = __import__('os')", "name '__import__'"),
        ("answer = getattr(1, 'real')", "name 'getattr'"),
        ("answer = (1).__class__", "attribute '__class__'"),
        ("answer = [].__class__.__base__.__subclasses__()", "attribute '__subclasses__'"),
        ("g = (x for x in [1])\nanswer = g.gi_frame", "attribute 'gi_frame'"),
        ("__builtins__ = {}", "name '__builtins__'"),
        ("class A:\n    pass", "ClassDef"),
        ("import math as __m", "alias '__m'"),
        ("def f(__x):\n    return 1", "argument name '__x'"),
        ("answer = (", "syntax error"),
    ],
)
def test_dangerous_code_is_rejected_before_execution(code: str, message: str) -> None:
    assert message in (find_code_violation(code, SETTINGS) or "")
    result = run_python(code, SETTINGS)
    assert result.status is ExecutionStatus.REJECTED
    assert result.answer is None


def test_code_length_limit() -> None:
    result = run_python("answer = 1\n" * 1000, replace(SETTINGS, max_code_chars=100))
    assert result.status is ExecutionStatus.REJECTED
    assert "limit" in (result.error or "")


def test_allowed_module_hides_its_imported_submodules() -> None:
    result = run_python("import fractions\nanswer = fractions.sys", SETTINGS)
    assert result.status is ExecutionStatus.ERROR
    assert "AttributeError" in (result.error or "")


def test_module_not_in_allowed_list_is_rejected() -> None:
    result = run_python("import decimal\nanswer = 1", replace(SETTINGS, allowed_modules=("math",)))
    assert result.status is ExecutionStatus.REJECTED


def test_infinite_loop_times_out() -> None:
    result = run_python("while True:\n    pass", FAST_TIMEOUT)
    assert result.status is ExecutionStatus.TIMEOUT


def test_memory_limit_is_enforced() -> None:
    result = run_python("data = [0] * (10 ** 9)", SETTINGS)
    assert result.status is ExecutionStatus.ERROR
    assert "MemoryError" in (result.error or "")


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("answer = 1 / 0", ExecutionStatus.ERROR),
        ("raise ValueError('погано')", ExecutionStatus.ERROR),
        ("x = 5", ExecutionStatus.NO_OUTPUT),
        ("answer = None", ExecutionStatus.NO_OUTPUT),
        ("answer = 'не знаю'", ExecutionStatus.UNPARSEABLE),
        ("answer = [1, 2]", ExecutionStatus.UNPARSEABLE),
        ("answer = float('nan')", ExecutionStatus.NO_OUTPUT),
    ],
)
def test_failed_or_empty_runs_have_no_answer(code: str, status: ExecutionStatus) -> None:
    result = run_python(code, SETTINGS)
    assert result.status is status
    assert result.answer is None


def test_output_is_truncated() -> None:
    result = run_python("print('1' * 10000)", replace(SETTINGS, max_output_chars=50))
    assert result.raw_output is not None
    assert len(result.raw_output) == 50


def test_solve_with_tool_without_code() -> None:
    result = solve_with_tool("Відповідь: 12", SETTINGS)
    assert result.status is ExecutionStatus.NO_CODE
    assert result.answer is None


def test_solve_batch_preserves_order_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    completions = [
        "```python\nanswer = 2 + 2\n```",
        "нема коду",
        "```python\nimport os\n```",
        "```python\nanswer = 9 * 9\n```",
    ]
    with caplog.at_level(logging.INFO):
        results = solve_batch_with_tool(completions, SETTINGS, max_workers=2)
    assert [result.answer for result in results] == ["4", None, None, "81"]
    assert [result.status for result in results] == [
        ExecutionStatus.OK,
        ExecutionStatus.NO_CODE,
        ExecutionStatus.REJECTED,
        ExecutionStatus.OK,
    ]
    assert "'ok': 2" in caplog.text


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"memory_limit_mb": 10}, "memory_limit_mb"),
        ({"max_code_chars": 0}, "max_code_chars"),
        ({"allowed_modules": ("math", "os")}, "unsafe modules"),
    ],
)
def test_sandbox_settings_validation(change: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        replace(SETTINGS, **change)
