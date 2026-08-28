import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from evidencebench.evaluation import (
    abstention_accuracy,
    citation_precision,
    recall_at_k,
    reciprocal_rank,
)


@dataclass(frozen=True)
class ApiEvaluationCase:
    case_id: str
    question: str
    relevant_chunk_ids: frozenset[str]
    should_abstain: bool


@dataclass(frozen=True)
class ApiEvaluationResult:
    case_id: str
    retrieved_chunk_ids: list[str]
    cited_chunk_ids: list[str]
    predicted_abstain: bool
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    citation_precision: float


def load_api_cases(path: Path) -> list[ApiEvaluationCase]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                raw["relevant_chunk_ids"] = frozenset(raw["relevant_chunk_ids"])
                cases.append(ApiEvaluationCase(**raw))
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid evaluation case at {path}:{line_number}") from exc
    if not cases:
        raise ValueError("Evaluation dataset is empty")
    return cases


async def evaluate_api(
    *, client: httpx.AsyncClient, cases: list[ApiEvaluationCase]
) -> dict[str, Any]:
    results = []
    for case in cases:
        response = await client.post(
            "/v1/query",
            json={"question": case.question, "include_retrieval_trace": True},
        )
        response.raise_for_status()
        payload = response.json()
        retrieved = [item["chunk_id"] for item in payload["retrieval_trace"]]
        cited = [item["chunk_id"] for item in payload["citations"]]
        results.append(
            ApiEvaluationResult(
                case_id=case.case_id,
                retrieved_chunk_ids=retrieved,
                cited_chunk_ids=cited,
                predicted_abstain=payload["verdict"] == "abstained",
                recall_at_5=recall_at_k(retrieved, case.relevant_chunk_ids, 5),
                recall_at_10=recall_at_k(retrieved, case.relevant_chunk_ids, 10),
                reciprocal_rank=reciprocal_rank(retrieved, case.relevant_chunk_ids),
                citation_precision=citation_precision(cited, case.relevant_chunk_ids),
            )
        )
    predictions = [result.predicted_abstain for result in results]
    expected = [case.should_abstain for case in cases]
    return {
        "schema_version": 1,
        "summary": {
            "case_count": len(cases),
            "recall_at_5": statistics.fmean(result.recall_at_5 for result in results),
            "recall_at_10": statistics.fmean(result.recall_at_10 for result in results),
            "mrr": statistics.fmean(result.reciprocal_rank for result in results),
            "citation_precision": statistics.fmean(result.citation_precision for result in results),
            "abstention_accuracy": abstention_accuracy(predictions, expected),
        },
        "results": [asdict(result) for result in results],
    }
