from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from math_solver.extract import DEFAULT_ANSWER_MARKER, ExtractionMethod, ExtractionResult, extract_answer

METHOD_PRIORITY: dict[ExtractionMethod, int] = {
    ExtractionMethod.MARKER: 0,
    ExtractionMethod.BOXED: 1,
    ExtractionMethod.LAST_NUMBER: 2,
    ExtractionMethod.FALLBACK: 3,
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoteResult:
    answer: str | None
    method: ExtractionMethod
    votes: int
    num_samples: int
    distribution: tuple[tuple[str, int], ...]

    @property
    def agreement(self) -> float:
        return self.votes / self.num_samples if self.num_samples else 0.0


@dataclass
class _Tally:
    first_index: int
    count: int = 0
    methods: Counter[ExtractionMethod] = field(default_factory=Counter)

    def rank_key(self) -> tuple[int, int, int, int]:
        return (
            -self.count,
            -self.methods[ExtractionMethod.MARKER],
            -self.methods[ExtractionMethod.BOXED],
            self.first_index,
        )

    def best_method(self) -> ExtractionMethod:
        return min(self.methods, key=METHOD_PRIORITY.__getitem__)


def vote_on_extractions(extractions: Sequence[ExtractionResult]) -> VoteResult:
    tallies: dict[str, _Tally] = {}
    for index, extraction in enumerate(extractions):
        if extraction.answer is None:
            continue
        tally = tallies.setdefault(extraction.answer, _Tally(first_index=index))
        tally.count += 1
        tally.methods[extraction.method] += 1
    if not tallies:
        return VoteResult(None, ExtractionMethod.FALLBACK, 0, len(extractions), ())
    ranked = sorted(tallies.items(), key=lambda item: item[1].rank_key())
    winner, tally = ranked[0]
    return VoteResult(
        answer=winner,
        method=tally.best_method(),
        votes=tally.count,
        num_samples=len(extractions),
        distribution=tuple((answer, item.count) for answer, item in ranked),
    )


def majority_vote(
    completions: Sequence[str],
    *,
    marker: str = DEFAULT_ANSWER_MARKER,
    last_number_fallback: bool = True,
) -> VoteResult:
    extractions = [
        extract_answer(completion, marker=marker, last_number_fallback=last_number_fallback)
        for completion in completions
    ]
    return vote_on_extractions(extractions)


def aggregate_self_consistency(
    completions_per_problem: Sequence[Sequence[str]],
    *,
    marker: str = DEFAULT_ANSWER_MARKER,
    last_number_fallback: bool = True,
    expected_samples: int | None = None,
) -> list[VoteResult]:
    if expected_samples is not None:
        wrong = [index for index, group in enumerate(completions_per_problem) if len(group) != expected_samples]
        if wrong:
            raise ValueError(f"{len(wrong)} problems do not have {expected_samples} samples, e.g. index {wrong[0]}")
    results = [
        majority_vote(group, marker=marker, last_number_fallback=last_number_fallback)
        for group in completions_per_problem
    ]
    log_agreement(results)
    return results


def log_agreement(results: Sequence[VoteResult]) -> None:
    if not results:
        logger.info("Self-consistency: no problems to aggregate")
        return
    mean_agreement = sum(result.agreement for result in results) / len(results)
    unanimous = sum(result.votes == result.num_samples and result.num_samples > 0 for result in results)
    unanswered = sum(result.answer is None for result in results)
    logger.info(
        "Self-consistency over %d problems: mean agreement %.3f, unanimous %d, no answer %d",
        len(results),
        mean_agreement,
        unanimous,
        unanswered,
    )
