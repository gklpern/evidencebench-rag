import pytest

from evidencebench.evaluation import (
    abstention_accuracy,
    citation_precision,
    recall_at_k,
    reciprocal_rank,
)


def test_retrieval_metrics() -> None:
    assert recall_at_k(["a", "b"], frozenset({"b", "c"}), 2) == 0.5
    assert reciprocal_rank(["a", "b"], frozenset({"b"})) == 0.5
    assert citation_precision(["a", "bad"], frozenset({"a"})) == 0.5


def test_abstention_accuracy() -> None:
    assert abstention_accuracy([True, False], [True, True]) == 0.5
    with pytest.raises(ValueError):
        abstention_accuracy([], [])
