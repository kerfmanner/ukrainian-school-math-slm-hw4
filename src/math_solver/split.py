from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from math_solver.data_io import require_columns

ID_COLUMN = "id"
MIN_STRATUM_SIZE = 2
SPLIT_FORMAT_VERSION = 1

logger = logging.getLogger(__name__)


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class SplitResult:
    seed: int
    val_ratio: float
    stratify_column: str
    source_rows: int
    source_fingerprint: str
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]


def fingerprint_frame(frame: pd.DataFrame, stratify_column: str) -> str:
    payload = frame[[ID_COLUMN, stratify_column]].astype(str).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_stratified_split(
    frame: pd.DataFrame,
    *,
    val_ratio: float,
    seed: int,
    stratify_column: str,
) -> SplitResult:
    _validate_source(frame, stratify_column)
    ids = frame[ID_COLUMN].astype(str)
    strata = frame[stratify_column].astype(str)
    rare_strata = _find_rare_strata(strata)
    eligible = ~strata.isin(rare_strata)
    try:
        _, val_part = train_test_split(
            ids[eligible],
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=strata[eligible],
        )
    except ValueError as error:
        raise SplitError(f"Stratified split failed: {error}") from error
    val_set = set(val_part)
    return SplitResult(
        seed=seed,
        val_ratio=val_ratio,
        stratify_column=stratify_column,
        source_rows=len(frame),
        source_fingerprint=fingerprint_frame(frame, stratify_column),
        train_ids=tuple(row_id for row_id in ids if row_id not in val_set),
        val_ids=tuple(row_id for row_id in ids if row_id in val_set),
    )


def apply_split(frame: pd.DataFrame, split: SplitResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_source(frame, split.stratify_column)
    if fingerprint_frame(frame, split.stratify_column) != split.source_fingerprint:
        raise SplitError("Source data does not match the data the split was created from")
    ids = frame[ID_COLUMN].astype(str)
    train = frame[ids.isin(set(split.train_ids))].reset_index(drop=True)
    val = frame[ids.isin(set(split.val_ids))].reset_index(drop=True)
    return train, val


def save_split(split: SplitResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": SPLIT_FORMAT_VERSION, **asdict(split)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_split(path: Path) -> SplitResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SplitError(f"Split file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise SplitError(f"Split file is not valid JSON: {path}") from error
    if payload.get("format_version") != SPLIT_FORMAT_VERSION:
        raise SplitError(f"Unsupported split format version in {path}: {payload.get('format_version')}")
    try:
        return SplitResult(
            seed=int(payload["seed"]),
            val_ratio=float(payload["val_ratio"]),
            stratify_column=str(payload["stratify_column"]),
            source_rows=int(payload["source_rows"]),
            source_fingerprint=str(payload["source_fingerprint"]),
            train_ids=tuple(str(row_id) for row_id in payload["train_ids"]),
            val_ids=tuple(str(row_id) for row_id in payload["val_ids"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SplitError(f"Split file {path} is malformed: {error}") from error


def _validate_source(frame: pd.DataFrame, stratify_column: str) -> None:
    require_columns(frame, [ID_COLUMN, stratify_column], "Split source")
    if frame.empty:
        raise SplitError("Split source is empty")
    for column in (ID_COLUMN, stratify_column):
        values = frame[column]
        if values.isna().any() or (values.astype(str).str.strip() == "").any():
            raise SplitError(f"Column '{column}' contains missing values")
    duplicated = frame[ID_COLUMN].astype(str).duplicated()
    if duplicated.any():
        raise SplitError(f"Column '{ID_COLUMN}' contains {int(duplicated.sum())} duplicate values")


def _find_rare_strata(strata: pd.Series) -> set[str]:
    counts = strata.value_counts()
    rare = set(counts[counts < MIN_STRATUM_SIZE].index)
    if rare:
        logger.warning(
            "%d strata have fewer than %d rows and are assigned to train: %s",
            len(rare),
            MIN_STRATUM_SIZE,
            sorted(rare),
        )
    return rare
