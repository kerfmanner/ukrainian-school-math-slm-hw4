from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import transformers
from packaging.version import Version
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

from math_solver.generation import (
    NOT_APPLICABLE,
    BackendInfo,
    FinishReason,
    Generation,
    GenerationError,
    Precision,
    SamplingConfig,
    resolve_precision,
)
from math_solver.prompts import ChatMessage

BACKEND_NAME = "hf"
END_OF_TURN_TOKEN = "<end_of_turn>"
BF16_MIN_COMPUTE_CAPABILITY = 8
GPU_INDEX = 0
NF4_QUANTIZATION = "bnb-nf4-double-quant"
BYTES_PER_GIGABYTE = 1024**3
DTYPE_KWARG_RENAME_VERSION = Version("4.56.0")

logger = logging.getLogger(__name__)


def native_bf16_supported() -> bool:
    major, _ = torch.cuda.get_device_capability(GPU_INDEX)
    if major < BF16_MIN_COMPUTE_CAPABILITY:
        return False
    # https://pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return bool(torch.cuda.is_bf16_supported())


class HFBackend:
    def __init__(self, *, model_id: str, adapter_path: Path | None, precision: Precision) -> None:
        cuda_available = torch.cuda.is_available()
        bf16_supported = cuda_available and native_bf16_supported()
        self._precision = resolve_precision(precision, cuda_available=cuda_available, bf16_supported=bf16_supported)
        self._tokenizer = _load_tokenizer(model_id)
        self._model = _load_model(model_id, self._precision)
        if adapter_path is not None:
            self._model = _attach_adapter(self._model, adapter_path)
        self._model.eval()
        self._pad_id = int(self._tokenizer.pad_token_id)
        self._stop_ids = _stop_token_ids(self._tokenizer, self._model.generation_config)
        self._info = BackendInfo(
            backend=BACKEND_NAME,
            model_id=model_id,
            adapter_path=None if adapter_path is None else str(adapter_path),
            precision=self._precision.value,
            dtype="bfloat16" if self._precision is Precision.BF16 else "float32",
            quantization=NOT_APPLICABLE if self._precision is Precision.BF16 else NF4_QUANTIZATION,
            device=f"cuda:{GPU_INDEX}",
            gpu_name=torch.cuda.get_device_name(GPU_INDEX),
        )
        logger.info(
            "Loaded %s on %s precision=%s stop_ids=%s allocated=%.2f GB",
            model_id,
            self._info.gpu_name,
            self._precision.value,
            sorted(self._stop_ids),
            torch.cuda.memory_allocated(GPU_INDEX) / BYTES_PER_GIGABYTE,
        )

    @property
    def info(self) -> BackendInfo:
        return self._info

    def format_prompt(self, messages: Sequence[ChatMessage]) -> str:
        return self._tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)

    def generate(self, prompts: Sequence[str], sampling: SamplingConfig, seed: int) -> list[list[Generation]]:
        transformers.set_seed(seed)
        encoded = self._tokenizer(list(prompts), return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = encoded.to(self._model.device)
        with torch.inference_mode():
            output = self._model.generate(**encoded, **self._generation_kwargs(sampling))
        new_tokens = output[:, encoded["input_ids"].shape[1] :].tolist()
        generations = [self._to_generation(tokens) for tokens in new_tokens]
        samples = sampling.num_samples
        return [generations[index * samples : (index + 1) * samples] for index in range(len(prompts))]

    def _generation_kwargs(self, sampling: SamplingConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": sampling.max_new_tokens,
            "num_return_sequences": sampling.num_samples,
            "do_sample": sampling.do_sample,
            "pad_token_id": self._pad_id,
            "eos_token_id": sorted(self._stop_ids),
        }
        if sampling.do_sample:
            kwargs.update(temperature=sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k)
        else:
            kwargs.update(temperature=None, top_p=None, top_k=None)
        return kwargs

    def _to_generation(self, tokens: list[int]) -> Generation:
        stop_index = next((index for index, token in enumerate(tokens) if token in self._stop_ids), None)
        body = tokens if stop_index is None else tokens[:stop_index]
        return Generation(
            text=self._tokenizer.decode(body, skip_special_tokens=True),
            num_tokens=len(body),
            finish_reason=FinishReason.LENGTH if stop_index is None else FinishReason.STOP,
            invalid_tokens=sum(token == self._pad_id for token in body),
        )


def _load_tokenizer(model_id: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.chat_template is None:
        raise GenerationError(f"Tokenizer for {model_id} has no chat template")
    if tokenizer.pad_token_id is None:
        raise GenerationError(f"Tokenizer for {model_id} has no pad token; batched left-padding is impossible")
    return tokenizer


def _load_model(model_id: str, precision: Precision) -> Any:
    model_class = _model_class(model_id)
    dtype_kwarg = "dtype" if Version(transformers.__version__) >= DTYPE_KWARG_RENAME_VERSION else "torch_dtype"
    kwargs: dict[str, Any] = {"device_map": {"": GPU_INDEX}}
    if precision is Precision.BF16:
        kwargs[dtype_kwarg] = torch.bfloat16
    else:
        kwargs[dtype_kwarg] = torch.float32
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float32,
        )
    return model_class.from_pretrained(model_id, **kwargs)


def _model_class(model_id: str) -> Any:
    architectures = getattr(AutoConfig.from_pretrained(model_id), "architectures", None) or []
    if not architectures:
        raise GenerationError(f"Config of {model_id} declares no architectures")
    model_class = getattr(transformers, architectures[0], None)
    if model_class is None:
        raise GenerationError(
            f"transformers {transformers.__version__} has no class '{architectures[0]}' needed by {model_id}"
        )
    return model_class


def _attach_adapter(model: Any, adapter_path: Path) -> Any:
    if not adapter_path.exists():
        raise GenerationError(f"Adapter path not found: {adapter_path}")
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter_path))


def _stop_token_ids(tokenizer: Any, generation_config: Any) -> frozenset[int]:
    configured = getattr(generation_config, "eos_token_id", None)
    configured_ids = configured if isinstance(configured, list) else [configured]
    candidates = {*configured_ids, tokenizer.eos_token_id}
    if END_OF_TURN_TOKEN in tokenizer.get_vocab():
        candidates.add(tokenizer.convert_tokens_to_ids(END_OF_TURN_TOKEN))
    stop_ids = frozenset(int(token) for token in candidates if token is not None)
    if not stop_ids:
        raise GenerationError("Could not determine any stop token ids")
    return stop_ids
