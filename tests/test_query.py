from collections.abc import Sequence

import pytest

from evidencebench.domain import (
    Candidate,
    Citation,
    FusedCandidate,
    GroundedAnswer,
    RetrievalChannel,
    Verdict,
)
from evidencebench.query import QueryService


class Embedder:
    dimensions = 2

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [(0.1, 0.2)]


class Repository:
    async def search(
        self,
        *,
        tenant_id: str,
        query: str,
        query_embedding: tuple[float, ...],
        channel: RetrievalChannel,
        limit: int,
    ) -> list[Candidate]:
        return [
            Candidate(
                "shared",
                "document",
                "The refund period is fourteen days.",
                "policy.pdf",
                4,
                1,
                0.9,
                channel,
            )
        ]


class Generator:
    def __init__(self, quote: str) -> None:
        self.quote = quote

    async def answer(self, question: str, evidence: list[FusedCandidate]) -> GroundedAnswer:
        return GroundedAnswer(
            "The period is fourteen days.",
            Verdict.ANSWERED,
            0.9,
            (Citation("c1", "shared", self.quote),),
        )


@pytest.mark.asyncio
async def test_query_returns_source_enriched_citation() -> None:
    service = QueryService(
        repository=Repository(),
        embedder=Embedder(),
        generator=Generator("fourteen days"),
        minimum_evidence=1,
    )
    answer = await service.query(tenant_id="tenant", question="What is the refund period?")
    assert answer.verdict == Verdict.ANSWERED
    assert answer.citations[0].source_uri == "policy.pdf"
    assert answer.citations[0].page == 4


@pytest.mark.asyncio
async def test_query_fails_closed_on_fabricated_quote() -> None:
    service = QueryService(
        repository=Repository(),
        embedder=Embedder(),
        generator=Generator("thirty days"),
        minimum_evidence=1,
    )
    answer = await service.query(tenant_id="tenant", question="What is the refund period?")
    assert answer.verdict == Verdict.ABSTAINED
    assert answer.reason and answer.reason.startswith("citation_validation_failed")
