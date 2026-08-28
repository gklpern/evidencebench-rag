from collections import defaultdict

from evidencebench.domain import Candidate, FusedCandidate, RetrievalChannel


def reciprocal_rank_fusion(
    rankings: dict[RetrievalChannel, list[Candidate]], *, rank_constant: int = 60, limit: int = 10
) -> list[FusedCandidate]:
    if rank_constant < 1 or limit < 1:
        raise ValueError("rank_constant and limit must be positive")
    scores: dict[str, float] = defaultdict(float)
    records: dict[str, Candidate] = {}
    channels: dict[str, set[RetrievalChannel]] = defaultdict(set)
    for channel, candidates in rankings.items():
        seen: set[str] = set()
        for position, candidate in enumerate(candidates, 1):
            if candidate.chunk_id in seen:
                continue
            seen.add(candidate.chunk_id)
            scores[candidate.chunk_id] += 1 / (rank_constant + position)
            channels[candidate.chunk_id].add(channel)
            records.setdefault(candidate.chunk_id, candidate)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
    return [
        FusedCandidate(
            chunk_id=chunk_id,
            document_id=records[chunk_id].document_id,
            content=records[chunk_id].content,
            source_uri=records[chunk_id].source_uri,
            page=records[chunk_id].page,
            score=scores[chunk_id],
            channels=tuple(sorted(channels[chunk_id], key=str)),
            metadata=records[chunk_id].metadata,
        )
        for chunk_id in ordered
    ]
