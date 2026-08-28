from dataclasses import dataclass, field
from enum import StrEnum


class RetrievalChannel(StrEnum):
    VECTOR = "vector"
    LEXICAL = "lexical"
    RERANK = "rerank"


class Verdict(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"


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


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    verdict: Verdict
    confidence: float
    citations: tuple[Citation, ...]
    reason: str | None = None
