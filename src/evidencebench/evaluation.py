from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    relevant_chunk_ids: frozenset[str]
    should_abstain: bool


def recall_at_k(retrieved: list[str], relevant: frozenset[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: frozenset[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, 1):
        if chunk_id in relevant:
            return 1 / rank
    return 0.0


def citation_precision(cited: list[str], supporting: frozenset[str]) -> float:
    if not cited:
        return 1.0 if not supporting else 0.0
    return sum(chunk_id in supporting for chunk_id in cited) / len(cited)


def abstention_accuracy(predictions: list[bool], expected: list[bool]) -> float:
    if len(predictions) != len(expected) or not expected:
        raise ValueError("prediction and expected lists must have the same non-zero length")
    return sum(
        prediction == truth for prediction, truth in zip(predictions, expected, strict=True)
    ) / len(expected)
