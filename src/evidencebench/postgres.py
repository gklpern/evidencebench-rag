import json
import math
from collections.abc import Sequence

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evidencebench.database import tenant_transaction
from evidencebench.domain import (
    Candidate,
    DocumentHandle,
    DocumentStatus,
    EmbeddedChunk,
    RetrievalChannel,
)


def vector_literal(vector: tuple[float, ...]) -> str:
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError("Embedding must contain finite values")
    return "[" + ",".join(format(value, ".9g") for value in vector) + "]"


class PostgresRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def begin_document(
        self,
        *,
        tenant_id: str,
        source_uri: str,
        sha256: str,
        title: str | None,
        metadata: dict[str, str],
    ) -> DocumentHandle:
        async with tenant_transaction(self.sessions, tenant_id) as session:
            inserted = await session.execute(
                text(
                    """
                    INSERT INTO documents (tenant_id, source_uri, sha256, title, status, metadata)
                    VALUES (CAST(:tenant_id AS uuid), :source_uri, :sha256, :title, 'pending',
                            CAST(:metadata AS jsonb))
                    ON CONFLICT (tenant_id, sha256) DO NOTHING
                    RETURNING id, status
                    """
                ),
                {
                    "tenant_id": tenant_id,
                    "source_uri": source_uri,
                    "sha256": sha256,
                    "title": title,
                    "metadata": json.dumps(metadata),
                },
            )
            row = inserted.mappings().one_or_none()
            if row:
                return DocumentHandle(str(row["id"]), DocumentStatus(row["status"]), True)
            existing = await session.execute(
                text(
                    "SELECT id, status FROM documents "
                    "WHERE tenant_id = CAST(:tenant_id AS uuid) AND sha256 = :sha256"
                ),
                {"tenant_id": tenant_id, "sha256": sha256},
            )
            row = existing.mappings().one()
            return DocumentHandle(str(row["id"]), DocumentStatus(row["status"]), False)

    async def mark_processing(self, *, tenant_id: str, document_id: str) -> None:
        await self._set_status(tenant_id, document_id, DocumentStatus.PROCESSING, None)

    async def fail_document(self, *, tenant_id: str, document_id: str, reason: str) -> None:
        await self._set_status(tenant_id, document_id, DocumentStatus.FAILED, reason)

    async def _set_status(
        self,
        tenant_id: str,
        document_id: str,
        status: DocumentStatus,
        reason: str | None,
    ) -> None:
        async with tenant_transaction(self.sessions, tenant_id) as session:
            result = await session.execute(
                text(
                    """
                    UPDATE documents
                    SET status = :status, error_message = :reason, updated_at = now()
                    WHERE id = CAST(:document_id AS uuid)
                    RETURNING id
                    """
                ),
                {"document_id": document_id, "status": status.value, "reason": reason},
            )
            if result.scalar_one_or_none() is None:
                raise LookupError("Document not found in tenant")

    async def complete_document(
        self, *, tenant_id: str, document_id: str, chunks: list[EmbeddedChunk]
    ) -> None:
        async with tenant_transaction(self.sessions, tenant_id) as session:
            await session.execute(
                text("DELETE FROM chunks WHERE document_id = CAST(:document_id AS uuid)"),
                {"document_id": document_id},
            )
            for chunk in chunks:
                await session.execute(
                    text(
                        """
                        INSERT INTO chunks (
                          tenant_id, document_id, ordinal, page, content, token_count,
                          embedding, metadata
                        ) VALUES (
                          CAST(:tenant_id AS uuid), CAST(:document_id AS uuid), :ordinal, :page,
                          :content, :token_count, CAST(:embedding AS vector),
                          CAST(:metadata AS jsonb)
                        )
                        """
                    ),
                    {
                        "tenant_id": tenant_id,
                        "document_id": document_id,
                        "ordinal": chunk.draft.ordinal,
                        "page": chunk.draft.page,
                        "content": chunk.draft.content,
                        "token_count": chunk.draft.token_count,
                        "embedding": vector_literal(chunk.embedding),
                        "metadata": json.dumps(chunk.draft.metadata),
                    },
                )
            result = await session.execute(
                text(
                    """
                    UPDATE documents
                    SET status = 'ready', error_message = NULL, updated_at = now()
                    WHERE id = CAST(:document_id AS uuid)
                    RETURNING id
                    """
                ),
                {"document_id": document_id},
            )
            if result.scalar_one_or_none() is None:
                raise LookupError("Document not found in tenant")

    async def search(
        self,
        *,
        tenant_id: str,
        query: str,
        query_embedding: tuple[float, ...],
        channel: RetrievalChannel,
        limit: int,
    ) -> list[Candidate]:
        if limit < 1:
            raise ValueError("limit must be positive")
        async with tenant_transaction(self.sessions, tenant_id) as session:
            if channel == RetrievalChannel.VECTOR:
                await session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
                rows = await session.execute(
                    text(
                        """
                        SELECT c.id, c.document_id, c.content, d.source_uri, c.page, c.metadata,
                               1 - (c.embedding <=> CAST(:embedding AS vector)) AS score
                        FROM chunks c JOIN documents d ON d.id = c.document_id
                        WHERE c.tenant_id = CAST(:tenant_id AS uuid) AND d.status = 'ready'
                        ORDER BY c.embedding <=> CAST(:embedding AS vector)
                        LIMIT :limit
                        """
                    ),
                    {
                        "tenant_id": tenant_id,
                        "embedding": vector_literal(query_embedding),
                        "limit": limit,
                    },
                )
            elif channel == RetrievalChannel.LEXICAL:
                rows = await session.execute(
                    text(
                        """
                        WITH q AS (SELECT websearch_to_tsquery('english', :query) AS value)
                        SELECT c.id, c.document_id, c.content, d.source_uri, c.page, c.metadata,
                               ts_rank_cd(c.search_vector, q.value, 32) AS score
                        FROM chunks c JOIN documents d ON d.id = c.document_id CROSS JOIN q
                        WHERE c.tenant_id = CAST(:tenant_id AS uuid) AND d.status = 'ready'
                          AND c.search_vector @@ q.value
                        ORDER BY score DESC, c.id
                        LIMIT :limit
                        """
                    ),
                    {"tenant_id": tenant_id, "query": query, "limit": limit},
                )
            else:
                raise ValueError(f"Unsupported retrieval channel: {channel}")
            return self._candidates(rows.mappings().all(), channel)

    @staticmethod
    def _candidates(rows: Sequence[RowMapping], channel: RetrievalChannel) -> list[Candidate]:
        return [
            Candidate(
                chunk_id=str(row["id"]),
                document_id=str(row["document_id"]),
                content=row["content"],
                source_uri=row["source_uri"],
                page=row["page"],
                rank=rank,
                score=float(row["score"]),
                channel=channel,
                metadata=dict(row["metadata"]),
            )
            for rank, row in enumerate(rows, 1)
        ]
