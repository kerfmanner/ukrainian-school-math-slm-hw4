from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from math_solver.generation import FinishReason, Generation
from math_solver.health import CompletionCheck

SIGNATURE_PARAM_KEYS: tuple[str, ...] = (
    "backend",
    "model_id",
    "adapter_path",
    "precision",
    "dtype",
    "quantization",
    "max_new_tokens",
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "num_samples",
    "seed",
)
SIGNATURE_RECORD_KEYS: tuple[str, ...] = ("split", "model_id", "prompt_name")
NEWLINE = b"\n"


class RunRecordError(ValueError):
    pass


@dataclass(frozen=True)
class CompletionStats:
    num_tokens: int
    finish_reason: FinishReason
    invalid_tokens: int = 0

    @classmethod
    def from_generation(cls, generation: Generation) -> CompletionStats:
        return cls(generation.num_tokens, generation.finish_reason, generation.invalid_tokens)

    def to_json(self) -> dict[str, Any]:
        return {
            "num_tokens": self.num_tokens,
            "finish_reason": self.finish_reason.value,
            "invalid_tokens": self.invalid_tokens,
        }


@dataclass(frozen=True)
class RunRecord:
    id: str
    skill: str
    split: str
    model_id: str
    prompt_name: str
    prompt_text: str
    completions: tuple[str, ...]
    completion_stats: tuple[CompletionStats, ...]
    generation_params: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.completions:
            raise RunRecordError(f"Record '{self.id}' has no completions")
        if len(self.completions) != len(self.completion_stats):
            raise RunRecordError(f"Record '{self.id}' has mismatched completions and completion_stats")

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "skill": self.skill,
            "split": self.split,
            "model_id": self.model_id,
            "prompt_name": self.prompt_name,
            "prompt_text": self.prompt_text,
            "completions": list(self.completions),
            "completion_stats": [stats.to_json() for stats in self.completion_stats],
            "generation_params": dict(self.generation_params),
        }

    def health_checks(self) -> list[CompletionCheck]:
        return [
            CompletionCheck(text, stats.finish_reason, stats.invalid_tokens)
            for text, stats in zip(self.completions, self.completion_stats)
        ]

    def signature(self) -> dict[str, Any]:
        return make_signature(self.split, self.model_id, self.prompt_name, self.generation_params)


@dataclass(frozen=True)
class RecordsFile:
    records: list[RunRecord]
    truncated_tail_offset: int | None = None


def make_signature(split: str, model_id: str, prompt_name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    record_part = dict(zip(SIGNATURE_RECORD_KEYS, (split, model_id, prompt_name)))
    params_part = {key: params.get(key) for key in SIGNATURE_PARAM_KEYS}
    return {**record_part, **params_part}


def read_records(path: Path) -> RecordsFile:
    if not path.exists():
        return RecordsFile([])
    data = path.read_bytes()
    lines = data.split(NEWLINE)
    tail = lines.pop()
    records: list[RunRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        record = _parse_line(line, f"{path}:{line_number}")
        if record.id in seen:
            raise RunRecordError(f"{path}:{line_number}: duplicate id '{record.id}'")
        seen.add(record.id)
        records.append(record)
    if tail.strip():
        return RecordsFile(records, truncated_tail_offset=len(data) - len(tail))
    return RecordsFile(records)


def repair_truncated_tail(path: Path, offset: int) -> None:
    with path.open("r+b") as handle:
        handle.truncate(offset)


def append_records(path: Path, records: Iterable[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record.to_json(), ensure_ascii=False) + "\n" for record in records)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def require_consistent_signature(records: Sequence[RunRecord], expected: Mapping[str, Any] | None = None) -> None:
    reference = dict(expected) if expected is not None else (records[0].signature() if records else None)
    for record in records:
        actual = record.signature()
        if actual != reference:
            differing = sorted(key for key in actual if actual.get(key) != reference.get(key))
            details = {key: (actual.get(key), reference.get(key)) for key in differing}
            raise RunRecordError(
                f"Record '{record.id}' was produced with different settings (record, expected): {details}"
            )


def _parse_line(line: bytes, location: str) -> RunRecord:
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunRecordError(f"{location}: invalid JSON line: {error}") from error
    if not isinstance(payload, dict):
        raise RunRecordError(f"{location}: each line must be a JSON object")
    try:
        return RunRecord(
            id=str(payload["id"]),
            skill=str(payload["skill"]),
            split=str(payload["split"]),
            model_id=str(payload["model_id"]),
            prompt_name=str(payload["prompt_name"]),
            prompt_text=str(payload["prompt_text"]),
            completions=tuple(str(text) for text in payload["completions"]),
            completion_stats=tuple(_parse_stats(item) for item in payload["completion_stats"]),
            generation_params=dict(payload["generation_params"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RunRecordError(f"{location}: malformed record: {error}") from error


def _parse_stats(item: Mapping[str, Any]) -> CompletionStats:
    return CompletionStats(
        num_tokens=int(item["num_tokens"]),
        finish_reason=FinishReason(item["finish_reason"]),
        invalid_tokens=int(item.get("invalid_tokens", 0)),
    )
