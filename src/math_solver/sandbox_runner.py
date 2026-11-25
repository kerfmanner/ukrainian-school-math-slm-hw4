from __future__ import annotations

import builtins
import contextlib
import importlib
import io
import json
import math
import sys
import types
from decimal import Decimal
from fractions import Fraction
from typing import Any

try:
    import resource
except ImportError:
    resource = None

SAFE_BUILTIN_NAMES: tuple[str, ...] = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "pow",
    "print",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "ArithmeticError",
    "Exception",
    "ValueError",
    "ZeroDivisionError",
)
FLOAT_SIGNIFICANT_DIGITS = 15
BYTES_PER_MEGABYTE = 1024 * 1024


def main() -> int:
    settings = json.loads(sys.argv[1])
    _apply_limits(settings["memory_limit_mb"], settings["cpu_seconds"])
    code = sys.stdin.buffer.read().decode("utf-8")
    output_limit = settings["max_output_chars"]
    buffer = io.StringIO()
    try:
        namespace = {"__builtins__": _safe_builtins(settings["allowed_modules"]), "__name__": "__sandbox__"}
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, "<model>", "exec"), namespace)
        payload = {"ok": True, "answer": _format_value(namespace.get("answer")), "stdout": buffer.getvalue()}
    except BaseException as error:
        payload = {"ok": False, "error": f"{type(error).__name__}: {error}", "stdout": buffer.getvalue()}
    payload = {key: value[:output_limit] if isinstance(value, str) else value for key, value in payload.items()}
    sys.stdout.write(json.dumps(payload, ensure_ascii=True))
    return 0


def _apply_limits(memory_limit_mb: int, cpu_seconds: int) -> None:
    if resource is None:
        return
    memory = memory_limit_mb * BYTES_PER_MEGABYTE
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


def _safe_builtins(allowed_modules: list[str]) -> dict[str, Any]:
    modules = {name: _public_namespace(importlib.import_module(name)) for name in allowed_modules}

    def restricted_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> types.SimpleNamespace:
        if level != 0 or name not in modules:
            raise ImportError(f"import of '{name}' is not allowed")
        return modules[name]

    safe = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
    safe["__import__"] = restricted_import
    return safe


def _public_namespace(module: types.ModuleType) -> types.SimpleNamespace:
    public = {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_") and not isinstance(value, types.ModuleType)
    }
    return types.SimpleNamespace(**public)


def _format_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, Decimal):
        return format(value, "f") if value.is_finite() else None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return format(Decimal(format(value, f".{FLOAT_SIGNIFICANT_DIGITS}g")), "f")
    return str(value).strip()


if __name__ == "__main__":
    sys.exit(main())
