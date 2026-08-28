from collections.abc import Sequence
from typing import Protocol

from evidencebench.domain import (
    Candidate,
    DocumentHandle,
    EmbeddedChunk,
    FusedCandidate,
    GroundedAnswer,
    RetrievalChannel,
)


class EmbeddingPort(Protocol):
    dimensions: int

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...


class GenerationPort(Protocol):
    async def answer(self, question: str, evidence: list[FusedCandidate]) -> GroundedAnswer: ...


class DocumentRepository(Protocol):
    async def begin_document(
        self,
        *,
        tenant_id: str,
        source_uri: str,
        sha256: str,
        title: str | None,
        metadata: dict[str, str],
    ) -> DocumentHandle: ...

    async def mark_processing(self, *, tenant_id: str, document_id: str) -> None: ...

    async def complete_document(
        self, *, tenant_id: str, document_id: str, chunks: list[EmbeddedChunk]
    ) -> None: ...

    async def fail_document(self, *, tenant_id: str, document_id: str, reason: str) -> None: ...


class RetrievalRepository(Protocol):
    async def search(
        self,
        *,
        tenant_id: str,
        query: str,
        query_embedding: tuple[float, ...],
        channel: RetrievalChannel,
        limit: int,
    ) -> list[Candidate]: ...
