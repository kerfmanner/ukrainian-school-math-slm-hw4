from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

StructuredT = TypeVar("StructuredT")


class ConfigError(ValueError):
    pass


def load_yaml_mapping(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except FileNotFoundError as error:
        raise ConfigError(f"File not found: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {path}: {error}") from error


def build_dataclass(cls: type[StructuredT], raw: Any, location: str = "") -> StructuredT:
    label = location or "root"
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'{label}' must be a mapping")
    known = {field.name: field for field in fields(cls)}
    required = {
        name for name, field in known.items() if field.default is MISSING and field.default_factory is MISSING
    }
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - set(known))
    if missing:
        raise ConfigError(f"'{label}' is missing keys: {missing}")
    if unknown:
        raise ConfigError(f"'{label}' has unknown keys: {unknown}")
    hints = get_type_hints(cls)
    values = {name: coerce_value(hints[name], raw[name], _join(location, name)) for name in known if name in raw}
    return cls(**values)


def coerce_value(target: Any, value: Any, location: str) -> Any:
    origin = get_origin(target)
    if origin in (Union, types.UnionType):
        return _coerce_optional(target, value, location)
    if is_dataclass(target):
        return build_dataclass(target, value, location)
    if target is bool:
        return _require_type(value, bool, location)
    if target is Path:
        return Path(_require_type(value, str, location))
    if target is str:
        return _require_type(value, str, location)
    if target is int:
        if isinstance(value, bool):
            raise ConfigError(f"'{location}' must be int, got bool")
        return _require_type(value, int, location)
    if target is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"'{location}' must be a number, got {type(value).__name__}")
        return float(value)
    if origin is dict:
        return _coerce_string_mapping(value, location)
    if origin is tuple:
        return _coerce_string_tuple(value, location)
    raise ConfigError(f"Unsupported type at '{location}': {target}")


def _coerce_optional(target: Any, value: Any, location: str) -> Any:
    options = [option for option in get_args(target) if option is not type(None)]
    if len(options) != 1:
        raise ConfigError(f"Unsupported union type at '{location}': {target}")
    if value is None:
        return None
    return coerce_value(options[0], value, location)


def _coerce_string_mapping(value: Any, location: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{location}' must be a mapping")
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        raise ConfigError(f"'{location}' must map strings to strings")
    return dict(value)


def _coerce_string_tuple(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"'{location}' must be a list of strings")
    return tuple(value)


def _require_type(value: Any, expected: type, location: str) -> Any:
    if not isinstance(value, expected):
        raise ConfigError(f"'{location}' must be {expected.__name__}, got {type(value).__name__}")
    return value


def _join(location: str, name: str) -> str:
    return f"{location}.{name}" if location else name
