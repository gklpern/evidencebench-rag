from collections.abc import Sequence

import pytest

from evidencebench.chunking import TextChunker, whitespace_token_count
from evidencebench.domain import DocumentHandle, DocumentStatus, EmbeddedChunk
from evidencebench.ingestion import IngestionService


class Embedder:
    dimensions = 2

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [(float(len(text)), 1.0) for text in texts]


class Repository:
    def __init__(self, *, existing: bool = False) -> None:
        self.existing = existing
        self.completed: list[EmbeddedChunk] = []
        self.failed: str | None = None

    async def begin_document(self, **kwargs: object) -> DocumentHandle:
        return DocumentHandle(
            "document",
            DocumentStatus.READY if self.existing else DocumentStatus.PENDING,
            not self.existing,
        )

    async def mark_processing(self, **kwargs: object) -> None:
        pass

    async def complete_document(
        self, *, tenant_id: str, document_id: str, chunks: list[EmbeddedChunk]
    ) -> None:
        self.completed = chunks

    async def fail_document(self, *, tenant_id: str, document_id: str, reason: str) -> None:
        self.failed = reason


@pytest.mark.asyncio
async def test_ingestion_embeds_and_commits_chunks() -> None:
    repository = Repository()
    service = IngestionService(
        repository=repository,
        embedder=Embedder(),
        chunker=TextChunker(token_counter=whitespace_token_count, max_tokens=3, overlap_tokens=0),
    )
    result = await service.ingest_text(
        tenant_id="tenant", source_uri="memory://doc", content="one two three four"
    )
    assert result.status == DocumentStatus.READY
    assert result.chunk_count == 2
    assert len(repository.completed) == 2


@pytest.mark.asyncio
async def test_ingestion_returns_existing_ready_document() -> None:
    repository = Repository(existing=True)
    service = IngestionService(
        repository=repository,
        embedder=Embedder(),
        chunker=TextChunker(token_counter=whitespace_token_count),
    )
    result = await service.ingest_text(
        tenant_id="tenant", source_uri="memory://doc", content="same content"
    )
    assert result.deduplicated is True
    assert repository.completed == []
