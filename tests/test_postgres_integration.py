import os
import uuid

import pytest

from evidencebench.database import create_database
from evidencebench.domain import ChunkDraft, EmbeddedChunk, RetrievalChannel
from evidencebench.postgres import PostgresRepository

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_real_postgres_ingestion_and_hybrid_search() -> None:
    database_url = os.environ.get("EB_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("EB_TEST_DATABASE_URL is not configured")
    database = create_database(database_url)
    repository = PostgresRepository(database.sessions)
    tenant_id = "00000000-0000-0000-0000-000000000001"
    document = await repository.begin_document(
        tenant_id=tenant_id,
        source_uri="integration://policy",
        sha256=uuid.uuid4().hex + uuid.uuid4().hex,
        title="Refund Policy",
        metadata={"suite": "integration"},
    )
    await repository.mark_processing(tenant_id=tenant_id, document_id=document.document_id)
    vector = (1.0, *([0.0] * 1023))
    await repository.complete_document(
        tenant_id=tenant_id,
        document_id=document.document_id,
        chunks=[
            EmbeddedChunk(
                ChunkDraft(0, "Refunds are available within fourteen days.", 6, page=3),
                vector,
            )
        ],
    )
    lexical = await repository.search(
        tenant_id=tenant_id,
        query="refund fourteen days",
        query_embedding=vector,
        channel=RetrievalChannel.LEXICAL,
        limit=5,
    )
    semantic = await repository.search(
        tenant_id=tenant_id,
        query="refund period",
        query_embedding=vector,
        channel=RetrievalChannel.VECTOR,
        limit=5,
    )
    await database.engine.dispose()
    assert lexical[0].document_id == document.document_id
    assert semantic[0].document_id == document.document_id
