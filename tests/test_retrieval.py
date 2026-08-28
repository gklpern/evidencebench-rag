from evidencebench.domain import Candidate, RetrievalChannel
from evidencebench.retrieval import reciprocal_rank_fusion


def candidate(chunk_id: str, rank: int, channel: RetrievalChannel) -> Candidate:
    return Candidate(chunk_id, "doc", chunk_id, "source", 1, rank, 1.0, channel)


def test_rrf_rewards_cross_channel_evidence() -> None:
    fused = reciprocal_rank_fusion(
        {
            RetrievalChannel.VECTOR: [
                candidate("vector-only", 1, RetrievalChannel.VECTOR),
                candidate("shared", 2, RetrievalChannel.VECTOR),
            ],
            RetrievalChannel.LEXICAL: [candidate("shared", 1, RetrievalChannel.LEXICAL)],
        }
    )
    assert fused[0].chunk_id == "shared"
    assert set(fused[0].channels) == {RetrievalChannel.VECTOR, RetrievalChannel.LEXICAL}
