# EvidenceBench RAG

An evidence-first enterprise retrieval platform designed to answer only when its sources are
strong enough. The system treats retrieval quality, citation integrity, tenant isolation and
abstention as measurable contracts rather than prompt-writing concerns.

## Design goals

- Tenant-isolated ingestion and retrieval with PostgreSQL row-level security
- Hybrid HNSW vector and full-text retrieval with reciprocal-rank fusion
- Optional cross-encoder reranking behind a stable interface
- Verbatim citation validation against immutable source chunks
- Explicit abstention when evidence coverage or confidence is insufficient
- Offline evaluation for recall@K, MRR, citation precision and abstention accuracy
- Raw evaluation artifacts suitable for regression gates in CI

## Retrieval path

```text
query ─┬─▶ pgvector HNSW ─┐
       └─▶ PostgreSQL FTS ─┴─▶ RRF ─▶ reranker ─▶ evidence gate
                                                        ├─▶ grounded answer + citations
                                                        └─▶ abstain + reason
```

The schema keeps `tenant_id` on both documents and chunks. Row-level policies use a
transaction-local tenant setting, and hybrid retrieval must apply the tenant predicate before
ranking. pgvector iterative scans are enabled by the repository layer when ANN filtering would
otherwise return too few tenant-local candidates.

## Local development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make install
cp .env.example .env
make test
make infra-up
```

## Current implementation

- Typed retrieval, citation and abstention domain contracts
- Deterministic reciprocal-rank fusion with duplicate suppression
- Citation quote/chunk validation
- Retrieval, citation and abstention metrics
- PostgreSQL 18 + pgvector schema with GIN, HNSW and row-level security
- Redis and S3-compatible object storage development stack
- Strict lint, type-check and test gates

Next: transactional repository layer, document ingestion state machine, chunking/embedding
workers, hybrid SQL retrieval, grounded-generation API, evaluation runner and observability.

## References

- [pgvector indexing, filtering and hybrid search](https://github.com/pgvector/pgvector)
- [PostgreSQL text-search functions](https://www.postgresql.org/docs/current/functions-textsearch.html)

