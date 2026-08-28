# EvidenceBench RAG

An evidence-first enterprise retrieval platform designed to answer only when its sources are
strong enough. Retrieval quality, citation integrity, tenant isolation and abstention are
measurable contracts rather than prompt-writing concerns.

## Design goals

- Tenant-isolated ingestion and retrieval with PostgreSQL row-level security
- Hybrid HNSW vector and full-text retrieval with reciprocal-rank fusion
- Verbatim citation validation against immutable source chunks
- Explicit abstention when evidence coverage or confidence is insufficient
- API-level evaluation for Recall@K, MRR, citation precision and abstention accuracy
- API keys mapped to tenant identities; clients cannot select tenant IDs directly

## Retrieval path

 s

```text
query ─┬─▶ pgvector HNSW ─┐
       └─▶ PostgreSQL FTS ─┴─▶ RRF ─▶ evidence gate
                                               ├─▶ grounded answer + verified citations
                                               └─▶ abstain + reason
```

The repository sets a transaction-local tenant identity before every operation. pgvector
iterative scans compensate for ANN filtering, while a composite foreign key prevents chunks
from being attached to documents owned by another tenant.

## Local development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make install
cp .env.example .env
make test
make infra-up
make serve
```

The development seed maps bearer token `dev-secret` to a fixed local tenant. Point
`EB_EMBEDDING_BASE_URL` and `EB_GENERATION_BASE_URL` at OpenAI-compatible local endpoints,
then ingest text:

```bash
curl http://127.0.0.1:8080/v1/documents/text \
  -H 'Authorization: Bearer dev-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "source_uri": "policy://refunds/2026",
    "title": "Refund Policy",
    "content": "Unused products may be returned within fourteen days."
  }'
```

Query with the same tenant credential:

```bash
curl http://127.0.0.1:8080/v1/query \
  -H 'Authorization: Bearer dev-secret' \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the refund period?"}'
```

Successful answers contain source URI, page when available, chunk ID and a quote verified
against the stored chunk. Invalid or insufficient evidence returns `verdict=abstained` with an
empty answer.

## Evaluation

Evaluation datasets are JSONL files containing questions, relevant chunk IDs and expected
abstention behavior. Run a suite against a deployed API:

```bash
EB_EVAL_API_KEY=dev-secret evidencebench evaluate \
  --dataset datasets/evaluation.example.jsonl \
  --output artifacts/evaluation.json
```

Artifacts include per-case traces plus Recall@5, Recall@10, MRR, citation precision and
abstention accuracy. Retrieval traces contain IDs and scores, not document text.

## Security model

- Bearer credentials resolve to server-side tenant IDs; payloads cannot choose a tenant.
- Every repository transaction sets `app.tenant_id` before accessing protected tables.
- PostgreSQL owner and API roles are separate; RLS is forced on tenant-scoped tables.
- Retrieved text is treated as untrusted data in the generation prompt.
- Generated quotes are checked against retrieved chunks; invalid citations fail closed.

## Implemented

- Idempotent SHA-256 text ingestion with explicit document state transitions
- Deterministic token-budget chunking with configurable overlap
- Batched, retried OpenAI-compatible embedding client with dimension validation
- Tenant-scoped transactional PostgreSQL repository
- Parallel vector/full-text retrieval with reciprocal-rank fusion
- Structured grounded-generation client with a prompt-injection boundary
- Evidence-based abstention and source-enriched citations
- Authenticated FastAPI ingestion/query endpoints, health checks and Prometheus metrics
- API-driven retrieval, citation and abstention evaluation runner
- PostgreSQL 18 + pgvector schema with GIN, HNSW and forced row-level security
- Redis/S3-compatible development services and a containerized API
- Strict lint, type-check, unit tests and a real-pgvector integration test in CI

Next: asynchronous ingestion jobs, PDF/DOCX/HTML parsers, object-store integration,
cross-encoder reranking, answer-faithfulness evaluation, load tests and Grafana dashboards.

## References

- [pgvector indexing, filtering and hybrid search](https://github.com/pgvector/pgvector)
- [PostgreSQL text-search functions](https://www.postgresql.org/docs/current/functions-textsearch.html)

