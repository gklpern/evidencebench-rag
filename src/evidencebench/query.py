import asyncio

from evidencebench.domain import Citation, GroundedAnswer, QueryOutcome, RetrievalChannel, Verdict
from evidencebench.grounding import should_abstain, validate_citations
from evidencebench.ports import EmbeddingPort, GenerationPort, RetrievalRepository
from evidencebench.retrieval import reciprocal_rank_fusion


class QueryService:
    def __init__(
        self,
        *,
        repository: RetrievalRepository,
        embedder: EmbeddingPort,
        generator: GenerationPort,
        per_channel_limit: int = 20,
        fused_limit: int = 8,
        minimum_evidence: int = 2,
        minimum_score: float = 0.015,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.generator = generator
        self.per_channel_limit = per_channel_limit
        self.fused_limit = fused_limit
        self.minimum_evidence = minimum_evidence
        self.minimum_score = minimum_score

    async def query(self, *, tenant_id: str, question: str) -> GroundedAnswer:
        return (await self.query_with_trace(tenant_id=tenant_id, question=question)).answer

    async def query_with_trace(self, *, tenant_id: str, question: str) -> QueryOutcome:
        vectors = await self.embedder.embed([question])
        if len(vectors) != 1:
            raise RuntimeError("Embedding service did not return exactly one query vector")
        vector_results, lexical_results = await asyncio.gather(
            self.repository.search(
                tenant_id=tenant_id,
                query=question,
                query_embedding=vectors[0],
                channel=RetrievalChannel.VECTOR,
                limit=self.per_channel_limit,
            ),
            self.repository.search(
                tenant_id=tenant_id,
                query=question,
                query_embedding=vectors[0],
                channel=RetrievalChannel.LEXICAL,
                limit=self.per_channel_limit,
            ),
        )
        evidence = reciprocal_rank_fusion(
            {
                RetrievalChannel.VECTOR: vector_results,
                RetrievalChannel.LEXICAL: lexical_results,
            },
            limit=self.fused_limit,
        )
        if should_abstain(
            evidence,
            minimum_evidence=self.minimum_evidence,
            minimum_score=self.minimum_score,
        ):
            return QueryOutcome(
                answer=GroundedAnswer(
                    answer="",
                    verdict=Verdict.ABSTAINED,
                    confidence=0,
                    citations=(),
                    reason="insufficient_evidence",
                ),
                evidence=tuple(evidence),
            )
        answer = await self.generator.answer(question, evidence)
        valid, errors = validate_citations(answer, evidence)
        if not valid:
            return QueryOutcome(
                answer=GroundedAnswer(
                    answer="",
                    verdict=Verdict.ABSTAINED,
                    confidence=0,
                    citations=(),
                    reason="citation_validation_failed: " + "; ".join(errors),
                ),
                evidence=tuple(evidence),
            )
        source_by_chunk = {candidate.chunk_id: candidate for candidate in evidence}
        return QueryOutcome(
            answer=GroundedAnswer(
                answer=answer.answer,
                verdict=answer.verdict,
                confidence=answer.confidence,
                citations=tuple(
                    Citation(
                        citation_id=citation.citation_id,
                        chunk_id=citation.chunk_id,
                        quote=citation.quote,
                        source_uri=source_by_chunk[citation.chunk_id].source_uri,
                        page=source_by_chunk[citation.chunk_id].page,
                    )
                    for citation in answer.citations
                ),
                reason=answer.reason,
            ),
            evidence=tuple(evidence),
        )
