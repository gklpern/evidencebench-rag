import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field
from sqlalchemy import text
from starlette.middleware.base import RequestResponseEndpoint

from evidencebench.auth import TenantAuthenticator
from evidencebench.chunking import TextChunker, whitespace_token_count
from evidencebench.clients import OpenAIEmbeddingClient, OpenAIGenerationClient
from evidencebench.database import Database, create_database
from evidencebench.domain import GroundedAnswer, IngestionResult, Verdict
from evidencebench.ingestion import IngestionService
from evidencebench.metrics import ABSTENTIONS, INGESTED_CHUNKS, LATENCY, REQUESTS
from evidencebench.postgres import PostgresRepository
from evidencebench.query import QueryService
from evidencebench.settings import get_settings


class TextDocumentRequest(BaseModel):
    source_uri: str = Field(min_length=1, max_length=2048)
    title: str | None = Field(default=None, max_length=500)
    content: str = Field(min_length=1, max_length=10_000_000)
    metadata: dict[str, str] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    include_retrieval_trace: bool = False


@dataclass(frozen=True)
class Runtime:
    database: Database
    embedding_http: httpx.AsyncClient
    generation_http: httpx.AsyncClient
    authenticator: TenantAuthenticator
    ingestion: IngestionService
    query: QueryService


def _headers(key: str) -> dict[str, str]:
    return {"authorization": f"Bearer {key}"} if key else {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    database = create_database(settings.database_url)
    embedding_http = httpx.AsyncClient(
        base_url=str(settings.embedding_base_url).rstrip("/"),
        headers=_headers(settings.embedding_api_key.get_secret_value()),
        timeout=60,
    )
    generation_http = httpx.AsyncClient(
        base_url=str(settings.generation_base_url).rstrip("/"),
        headers=_headers(settings.generation_api_key.get_secret_value()),
        timeout=120,
    )
    repository = PostgresRepository(database.sessions)
    embedder = OpenAIEmbeddingClient(
        client=embedding_http,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    app.state.runtime = Runtime(
        database=database,
        embedding_http=embedding_http,
        generation_http=generation_http,
        authenticator=TenantAuthenticator(settings.api_keys_json.get_secret_value()),
        ingestion=IngestionService(
            repository=repository,
            embedder=embedder,
            chunker=TextChunker(token_counter=whitespace_token_count),
        ),
        query=QueryService(
            repository=repository,
            embedder=embedder,
            generator=OpenAIGenerationClient(
                client=generation_http, model=settings.generation_model
            ),
            per_channel_limit=settings.retrieval_limit_per_channel,
            fused_limit=settings.fused_limit,
            minimum_evidence=settings.minimum_evidence,
            minimum_score=settings.minimum_rrf_score,
        ),
    )
    yield
    await embedding_http.aclose()
    await generation_http.aclose()
    await database.engine.dispose()


app = FastAPI(title="EvidenceBench RAG", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def instrumentation(request: Request, call_next: RequestResponseEndpoint) -> Response:
    started = time.perf_counter()
    route = request.url.path
    try:
        response = await call_next(request)
        REQUESTS.labels(route, str(response.status_code)).inc()
        response.headers["x-request-id"] = request.headers.get("x-request-id", str(uuid.uuid4()))
        return response
    except Exception:
        REQUESTS.labels(route, "500").inc()
        raise
    finally:
        LATENCY.labels(route).observe(time.perf_counter() - started)


def tenant_id(request: Request, authorization: Annotated[str | None, Header()] = None) -> str:
    runtime: Runtime = request.app.state.runtime
    tenant = runtime.authenticator.authenticate(authorization)
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return tenant


@app.get("/health/live", include_in_schema=False)
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready", include_in_schema=False)
async def ready(request: Request) -> dict[str, str]:
    runtime: Runtime = request.app.state.runtime
    try:
        async with runtime.database.sessions() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return {"status": "ready"}


@app.post("/v1/documents/text", status_code=201)
async def ingest_document(
    payload: TextDocumentRequest,
    request: Request,
    tenant: Annotated[str, Depends(tenant_id)],
) -> dict[str, Any]:
    runtime: Runtime = request.app.state.runtime
    result: IngestionResult = await runtime.ingestion.ingest_text(
        tenant_id=tenant,
        source_uri=payload.source_uri,
        title=payload.title,
        content=payload.content,
        metadata=payload.metadata,
    )
    INGESTED_CHUNKS.inc(result.chunk_count)
    return asdict(result)


@app.post("/v1/query")
async def query(
    payload: QueryRequest,
    request: Request,
    tenant: Annotated[str, Depends(tenant_id)],
) -> dict[str, Any]:
    runtime: Runtime = request.app.state.runtime
    outcome = await runtime.query.query_with_trace(tenant_id=tenant, question=payload.question)
    result: GroundedAnswer = outcome.answer
    if result.verdict == Verdict.ABSTAINED:
        ABSTENTIONS.labels(result.reason or "unspecified").inc()
    response = asdict(result)
    if payload.include_retrieval_trace:
        response["retrieval_trace"] = [
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "score": item.score,
                "channels": [channel.value for channel in item.channels],
            }
            for item in outcome.evidence
        ]
    return response
