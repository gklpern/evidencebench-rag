import hashlib

from evidencebench.chunking import TextChunker
from evidencebench.domain import DocumentStatus, EmbeddedChunk, IngestionResult
from evidencebench.ports import DocumentRepository, EmbeddingPort


class IngestionService:
    def __init__(
        self,
        *,
        repository: DocumentRepository,
        embedder: EmbeddingPort,
        chunker: TextChunker,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.chunker = chunker

    async def ingest_text(
        self,
        *,
        tenant_id: str,
        source_uri: str,
        content: str,
        title: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> IngestionResult:
        encoded = content.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        handle = await self.repository.begin_document(
            tenant_id=tenant_id,
            source_uri=source_uri,
            sha256=digest,
            title=title,
            metadata=metadata or {},
        )
        if not handle.created and handle.status == DocumentStatus.READY:
            return IngestionResult(handle.document_id, handle.status, 0, True)
        await self.repository.mark_processing(tenant_id=tenant_id, document_id=handle.document_id)
        try:
            drafts = self.chunker.split(content)
            if not drafts:
                raise ValueError("Document contains no indexable text")
            vectors = await self.embedder.embed([draft.content for draft in drafts])
            if len(vectors) != len(drafts):
                raise RuntimeError("Embedding service returned a mismatched vector count")
            if any(len(vector) != self.embedder.dimensions for vector in vectors):
                raise RuntimeError("Embedding dimensions do not match the configured schema")
            chunks = [
                EmbeddedChunk(draft=draft, embedding=vector)
                for draft, vector in zip(drafts, vectors, strict=True)
            ]
            await self.repository.complete_document(
                tenant_id=tenant_id, document_id=handle.document_id, chunks=chunks
            )
            return IngestionResult(handle.document_id, DocumentStatus.READY, len(chunks), False)
        except Exception as exc:
            await self.repository.fail_document(
                tenant_id=tenant_id, document_id=handle.document_id, reason=str(exc)[:1000]
            )
            raise
