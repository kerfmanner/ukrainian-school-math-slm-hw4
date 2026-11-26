from __future__ import annotations

import pytest

from math_solver.generation import (
    GenerationError,
    MockBackend,
    Precision,
    SamplingConfig,
    create_backend,
    derive_batch_seed,
    resolve_precision,
)

GREEDY = SamplingConfig(max_new_tokens=16, temperature=0.0, top_p=1.0, top_k=0, num_samples=1)


@pytest.mark.parametrize(
    ("requested", "bf16_supported", "expected"),
    [
        (Precision.AUTO, True, Precision.BF16),
        (Precision.AUTO, False, Precision.FOUR_BIT_FP32),
        (Precision.BF16, True, Precision.BF16),
        (Precision.FOUR_BIT_FP32, True, Precision.FOUR_BIT_FP32),
        (Precision.FOUR_BIT_FP32, False, Precision.FOUR_BIT_FP32),
    ],
)
def test_resolve_precision(requested: Precision, bf16_supported: bool, expected: Precision) -> None:
    assert resolve_precision(requested, cuda_available=True, bf16_supported=bf16_supported) is expected


def test_explicit_bf16_on_unsupported_gpu_raises() -> None:
    with pytest.raises(GenerationError, match="no native bfloat16"):
        resolve_precision(Precision.BF16, cuda_available=True, bf16_supported=False)


def test_precision_without_cuda_raises() -> None:
    with pytest.raises(GenerationError, match="No CUDA GPU"):
        resolve_precision(Precision.AUTO, cuda_available=False, bf16_supported=False)


def test_multiple_greedy_samples_are_rejected() -> None:
    with pytest.raises(ValueError, match="temperature > 0"):
        SamplingConfig(max_new_tokens=16, temperature=0.0, top_p=1.0, top_k=0, num_samples=3)


@pytest.mark.parametrize(
    "overrides",
    [{"max_new_tokens": 0}, {"num_samples": 0}, {"temperature": -0.1}, {"top_p": 0.0}, {"top_k": -1}],
)
def test_invalid_sampling_values_raise(overrides: dict[str, float]) -> None:
    values = {"max_new_tokens": 16, "temperature": 0.7, "top_p": 1.0, "top_k": 0, "num_samples": 1, **overrides}
    with pytest.raises(ValueError):
        SamplingConfig(**values)


def test_mock_backend_returns_requested_samples_per_prompt() -> None:
    backend = MockBackend(model_id="google/gemma-3-4b-it")
    sampling = SamplingConfig(max_new_tokens=16, temperature=0.7, top_p=1.0, top_k=0, num_samples=3)
    outputs = backend.generate(["a", "b"], sampling, seed=7)
    assert [len(group) for group in outputs] == [3, 3]
    assert backend.batch_sizes == [2]
    assert backend.seeds == [7]
    assert "Відповідь:" in outputs[0][0].text


def test_mock_backend_formats_chat_messages() -> None:
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    prompt = MockBackend(model_id="m").format_prompt(messages)
    assert prompt.index("S") < prompt.index("U")


def test_batch_seed_is_deterministic_and_depends_on_ids() -> None:
    assert derive_batch_seed(42, ["a", "b"]) == derive_batch_seed(42, ["a", "b"])
    assert derive_batch_seed(42, ["a", "b"]) != derive_batch_seed(42, ["a", "c"])
    assert derive_batch_seed(42, ["a"]) != derive_batch_seed(43, ["a"])


def test_create_backend() -> None:
    backend = create_backend("mock", model_id="m", adapter_path=None, precision=Precision.AUTO)
    assert backend.info.backend == "mock"
    with pytest.raises(GenerationError, match="Unknown backend"):
        create_backend("vllm", model_id="m", adapter_path=None, precision=Precision.AUTO)
