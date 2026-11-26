from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from math_solver.prompts import ChatMessage

SEED_HEX_DIGITS = 8
MOCK_BACKEND = "mock"
HF_BACKEND = "hf"
BACKENDS: tuple[str, ...] = (HF_BACKEND, MOCK_BACKEND)
NOT_APPLICABLE = "none"


class GenerationError(RuntimeError):
    pass


class Precision(StrEnum):
    AUTO = "auto"
    BF16 = "bf16"
    FOUR_BIT_FP32 = "4bit-fp32"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    num_samples: int

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.num_samples > 1 and not self.do_sample:
            raise ValueError(f"num_samples={self.num_samples} requires temperature > 0; greedy samples are identical")

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0


@dataclass(frozen=True)
class Generation:
    text: str
    num_tokens: int
    finish_reason: FinishReason
    invalid_tokens: int = 0


@dataclass(frozen=True)
class BackendInfo:
    backend: str
    model_id: str
    adapter_path: str | None
    precision: str
    dtype: str
    quantization: str
    device: str
    gpu_name: str | None


class GeneratorBackend(Protocol):
    @property
    def info(self) -> BackendInfo: ...

    def format_prompt(self, messages: Sequence[ChatMessage]) -> str: ...

    def generate(self, prompts: Sequence[str], sampling: SamplingConfig, seed: int) -> list[list[Generation]]: ...


Responder = Callable[[str, int], str]


def mock_responder(prompt: str, sample_index: int) -> str:
    value = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:SEED_HEX_DIGITS], 16) % 100 + sample_index
    return f"Обчислення.\n```python\nanswer = {value}\n```\nВідповідь: {value}"


@dataclass
class MockBackend:
    model_id: str
    adapter_path: str | None = None
    responder: Responder = mock_responder
    batch_sizes: list[int] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            backend=MOCK_BACKEND,
            model_id=self.model_id,
            adapter_path=self.adapter_path,
            precision=NOT_APPLICABLE,
            dtype=NOT_APPLICABLE,
            quantization=NOT_APPLICABLE,
            device="cpu",
            gpu_name=None,
        )

    def format_prompt(self, messages: Sequence[ChatMessage]) -> str:
        turns = [f"<{message['role']}>\n{message['content']}" for message in messages]
        return "\n".join(turns) + "\n<model>\n"

    def generate(self, prompts: Sequence[str], sampling: SamplingConfig, seed: int) -> list[list[Generation]]:
        self.batch_sizes.append(len(prompts))
        self.seeds.append(seed)
        return [
            [self._generation(self.responder(prompt, index)) for index in range(sampling.num_samples)]
            for prompt in prompts
        ]

    @staticmethod
    def _generation(text: str) -> Generation:
        return Generation(text=text, num_tokens=len(text.split()), finish_reason=FinishReason.STOP)


def resolve_precision(requested: Precision, *, cuda_available: bool, bf16_supported: bool) -> Precision:
    if not cuda_available:
        raise GenerationError("No CUDA GPU detected; the hf backend needs a GPU (use --backend mock for CPU tests)")
    if requested is Precision.AUTO:
        return Precision.BF16 if bf16_supported else Precision.FOUR_BIT_FP32
    if requested is Precision.BF16 and not bf16_supported:
        raise GenerationError(
            "--precision bf16 was requested, but this GPU has no native bfloat16 support (e.g. T4); "
            "use --precision auto or --precision 4bit-fp32"
        )
    return requested


def derive_batch_seed(seed: int, ids: Sequence[str]) -> int:
    digest = hashlib.sha256(f"{seed}:{','.join(ids)}".encode("utf-8")).hexdigest()
    return int(digest[:SEED_HEX_DIGITS], 16)


def create_backend(
    backend: str,
    *,
    model_id: str,
    adapter_path: Path | None,
    precision: Precision,
) -> GeneratorBackend:
    adapter = None if adapter_path is None else str(adapter_path)
    if backend == MOCK_BACKEND:
        return MockBackend(model_id=model_id, adapter_path=adapter)
    if backend == HF_BACKEND:
        try:
            from math_solver.hf_backend import HFBackend
        except ImportError as error:
            raise GenerationError(f"The hf backend needs torch and the gpu extra installed: {error}") from error
        return HFBackend(model_id=model_id, adapter_path=adapter_path, precision=precision)
    raise GenerationError(f"Unknown backend '{backend}'; expected one of {list(BACKENDS)}")
