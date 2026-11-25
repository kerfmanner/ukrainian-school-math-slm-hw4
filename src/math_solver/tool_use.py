from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from math_solver.normalize import normalize_answer
from math_solver.structured import ConfigError

RUNNER_PATH = Path(__file__).with_name("sandbox_runner.py")
SAFE_MODULES = frozenset({"math", "fractions", "decimal"})
PYTHON_LANGUAGES = frozenset({"python", "python3", "py"})
BLOCKED_NAMES = frozenset(
    {
        "breakpoint",
        "classmethod",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "memoryview",
        "object",
        "open",
        "property",
        "quit",
        "setattr",
        "staticmethod",
        "super",
        "type",
        "vars",
    }
)
BLOCKED_ATTRIBUTE_PREFIXES: tuple[str, ...] = ("_", "f_", "gi_", "cr_", "ag_", "tb_", "co_")
BLOCKED_NODES: tuple[type[ast.AST], ...] = (ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Global, ast.Nonlocal)
CODE_BLOCK_PATTERN = re.compile(r"```[ \t]*(?P<language>[A-Za-z0-9_+-]*)[ \t]*\r?\n(?P<code>.*?)```", re.DOTALL)
MAX_ERROR_CHARS = 300

logger = logging.getLogger(__name__)


class ExecutionStatus(StrEnum):
    OK = "ok"
    NO_CODE = "no_code"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    ERROR = "error"
    NO_OUTPUT = "no_output"
    UNPARSEABLE = "unparseable"


@dataclass(frozen=True)
class SandboxSettings:
    timeout_seconds: float
    memory_limit_mb: int
    max_code_chars: int
    max_output_chars: int
    allowed_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ConfigError(f"sandbox.timeout_seconds must be positive, got {self.timeout_seconds}")
        if self.memory_limit_mb < 64:
            raise ConfigError(f"sandbox.memory_limit_mb must be >= 64, got {self.memory_limit_mb}")
        if self.max_code_chars <= 0 or self.max_output_chars <= 0:
            raise ConfigError("sandbox.max_code_chars and sandbox.max_output_chars must be positive")
        unsafe = sorted(set(self.allowed_modules) - SAFE_MODULES)
        if unsafe:
            raise ConfigError(f"sandbox.allowed_modules contains unsafe modules {unsafe}; allowed: {sorted(SAFE_MODULES)}")


@dataclass(frozen=True)
class ToolExecutionResult:
    status: ExecutionStatus
    answer: str | None
    raw_output: str | None = None
    error: str | None = None


def extract_python_code(completion: str) -> str | None:
    blocks = list(CODE_BLOCK_PATTERN.finditer(completion))
    python_blocks = [block for block in blocks if block["language"].lower() in PYTHON_LANGUAGES]
    unlabeled_blocks = [block for block in blocks if not block["language"]]
    chosen = python_blocks or unlabeled_blocks
    if not chosen:
        return None
    code = chosen[-1]["code"].strip()
    return code or None


def find_code_violation(code: str, settings: SandboxSettings) -> str | None:
    if len(code) > settings.max_code_chars:
        return f"code has {len(code)} characters, limit is {settings.max_code_chars}"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        return f"syntax error: {error.msg} (line {error.lineno})"
    allowed = set(settings.allowed_modules)
    for node in ast.walk(tree):
        violation = _node_violation(node, allowed)
        if violation is not None:
            return violation
    return None


def run_python(code: str, settings: SandboxSettings) -> ToolExecutionResult:
    violation = find_code_violation(code, settings)
    if violation is not None:
        return ToolExecutionResult(ExecutionStatus.REJECTED, None, error=violation)
    runner_settings = {
        "allowed_modules": list(settings.allowed_modules),
        "memory_limit_mb": settings.memory_limit_mb,
        "cpu_seconds": max(1, math.ceil(settings.timeout_seconds)),
        "max_output_chars": settings.max_output_chars,
    }
    with tempfile.TemporaryDirectory() as workdir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-B", str(RUNNER_PATH), json.dumps(runner_settings)],
                input=code.encode("utf-8"),
                capture_output=True,
                timeout=settings.timeout_seconds,
                cwd=workdir,
                env={},
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolExecutionResult(
                ExecutionStatus.TIMEOUT, None, error=f"timed out after {settings.timeout_seconds}s"
            )
    return _interpret(completed)


def solve_with_tool(completion: str, settings: SandboxSettings) -> ToolExecutionResult:
    code = extract_python_code(completion)
    if code is None:
        return ToolExecutionResult(ExecutionStatus.NO_CODE, None)
    return run_python(code, settings)


def solve_batch_with_tool(
    completions: Sequence[str],
    settings: SandboxSettings,
    max_workers: int | None = None,
) -> list[ToolExecutionResult]:
    workers = max_workers or min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda completion: solve_with_tool(completion, settings), completions))
    counts = Counter(result.status.value for result in results)
    logger.info("Tool execution statuses over %d completions: %s", len(results), dict(sorted(counts.items())))
    return results


def _node_violation(node: ast.AST, allowed_modules: set[str]) -> str | None:
    if isinstance(node, BLOCKED_NODES):
        return f"'{type(node).__name__}' statements are not allowed"
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name not in allowed_modules:
                return f"import of '{alias.name}' is not allowed"
            if alias.asname is not None and alias.asname.startswith("__"):
                return f"alias '{alias.asname}' is not allowed"
    if isinstance(node, ast.ImportFrom):
        if node.level or node.module not in allowed_modules:
            return f"import from '{node.module}' is not allowed"
        if any(alias.name == "*" or alias.name.startswith("_") for alias in node.names):
            return f"import of private or wildcard names from '{node.module}' is not allowed"
    if isinstance(node, ast.Attribute) and node.attr.startswith(BLOCKED_ATTRIBUTE_PREFIXES):
        return f"attribute '{node.attr}' is not allowed"
    if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in BLOCKED_NAMES):
        return f"name '{node.id}' is not allowed"
    if isinstance(node, ast.FunctionDef) and node.name.startswith("__"):
        return f"function name '{node.name}' is not allowed"
    if isinstance(node, ast.arg) and node.arg.startswith("__"):
        return f"argument name '{node.arg}' is not allowed"
    if isinstance(node, ast.ExceptHandler) and node.name is not None and node.name.startswith("__"):
        return f"exception name '{node.name}' is not allowed"
    return None


def _interpret(completed: subprocess.CompletedProcess[bytes]) -> ToolExecutionResult:
    stdout = completed.stdout.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0 or not stdout:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        detail = stderr[-MAX_ERROR_CHARS:] if stderr else f"exit code {completed.returncode}"
        return ToolExecutionResult(ExecutionStatus.ERROR, None, error=detail)
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return ToolExecutionResult(ExecutionStatus.ERROR, None, error="sandbox produced malformed output")
    if not payload.get("ok"):
        return ToolExecutionResult(ExecutionStatus.ERROR, None, error=str(payload.get("error"))[:MAX_ERROR_CHARS])
    raw_output = payload.get("answer") or _last_non_empty_line(str(payload.get("stdout", "")))
    if raw_output is None:
        return ToolExecutionResult(ExecutionStatus.NO_OUTPUT, None)
    answer = normalize_answer(raw_output)
    if answer is None:
        return ToolExecutionResult(ExecutionStatus.UNPARSEABLE, None, raw_output=raw_output)
    return ToolExecutionResult(ExecutionStatus.OK, answer, raw_output=raw_output)


def _last_non_empty_line(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None
