import httpx
import pytest

from evidencebench.evaluation_runner import ApiEvaluationCase, evaluate_api


@pytest.mark.asyncio
async def test_api_evaluator_aggregates_metrics() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "verdict": "answered",
                "citations": [{"chunk_id": "relevant"}],
                "retrieval_trace": [
                    {"chunk_id": "other"},
                    {"chunk_id": "relevant"},
                ],
            },
        )

    cases = [ApiEvaluationCase("case", "question", frozenset({"relevant"}), should_abstain=False)]
    async with httpx.AsyncClient(
        base_url="http://api", transport=httpx.MockTransport(handler)
    ) as client:
        result = await evaluate_api(client=client, cases=cases)
    assert result["summary"] == {
        "case_count": 1,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "mrr": 0.5,
        "citation_precision": 1.0,
        "abstention_accuracy": 1.0,
    }
