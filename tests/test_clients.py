import json

import httpx
import pytest

from evidencebench.clients import OpenAIEmbeddingClient, OpenAIGenerationClient
from evidencebench.domain import FusedCandidate, RetrievalChannel, Verdict


@pytest.mark.asyncio
async def test_embedding_client_preserves_server_order() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embeddings"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [3, 4]},
                    {"index": 0, "embedding": [1, 2]},
                ]
            },
        )

    async with httpx.AsyncClient(
        base_url="http://embed", transport=httpx.MockTransport(handler)
    ) as http:
        client = OpenAIEmbeddingClient(client=http, model="embed", dimensions=2)
        assert await client.embed(["first", "second"]) == [(1.0, 2.0), (3.0, 4.0)]


@pytest.mark.asyncio
async def test_generation_client_parses_grounded_contract() -> None:
    content = json.dumps(
        {
            "answer": "Fourteen days.",
            "verdict": "answered",
            "confidence": 0.9,
            "citations": [{"chunk_id": "chunk", "quote": "fourteen days"}],
            "reason": None,
        }
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["temperature"] == 0
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    evidence = [
        FusedCandidate(
            "chunk",
            "document",
            "The period is fourteen days.",
            "policy.pdf",
            1,
            0.03,
            (RetrievalChannel.VECTOR,),
        )
    ]
    async with httpx.AsyncClient(
        base_url="http://generate", transport=httpx.MockTransport(handler)
    ) as http:
        answer = await OpenAIGenerationClient(client=http, model="model").answer(
            "What is the period?", evidence
        )
    assert answer.verdict == Verdict.ANSWERED
    assert answer.citations[0].chunk_id == "chunk"
