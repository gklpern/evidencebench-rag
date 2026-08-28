from dataclasses import dataclass, field
from enum import StrEnum


class RetrievalChannel(StrEnum):
    VECTOR = "vector"
    LEXICAL = "lexical"
    RERANK = "rerank"


class Verdict(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    document_id: str
    content: str
    source_uri: str
    page: int | None
    rank: int
    score: float
    channel: RetrievalChannel
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FusedCandidate:
    chunk_id: str
    document_id: str
    content: str
    source_uri: str
    page: int | None
    score: float
    channels: tuple[RetrievalChannel, ...]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    citation_id: str
    chunk_id: str
    quote: str
    source_uri: str | None = None
    page: int | None = None


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    verdict: Verdict
    confidence: float
    citations: tuple[Citation, ...]
    reason: str | None = None


@dataclass(frozen=True)
class ChunkDraft:
    ordinal: int
    content: str
    token_count: int
    page: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddedChunk:
    draft: ChunkDraft
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class DocumentHandle:
    document_id: str
    status: DocumentStatus
    created: bool


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    status: DocumentStatus
    chunk_count: int
    deduplicated: bool


@dataclass(frozen=True)
class QueryOutcome:
    answer: GroundedAnswer
    evidence: tuple[FusedCandidate, ...]
