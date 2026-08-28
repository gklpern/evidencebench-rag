from evidencebench.domain import Citation, FusedCandidate, GroundedAnswer, RetrievalChannel, Verdict
from evidencebench.grounding import should_abstain, validate_citations


def evidence(score: float = 0.1) -> FusedCandidate:
    return FusedCandidate(
        "chunk",
        "doc",
        "Revenue increased by 12 percent.",
        "uri",
        2,
        score,
        (RetrievalChannel.VECTOR,),
    )


def test_citation_quote_must_exist_in_evidence() -> None:
    answer = GroundedAnswer(
        "Revenue grew.",
        Verdict.ANSWERED,
        0.9,
        (Citation("c1", "chunk", "increased by 12 percent"),),
    )
    assert validate_citations(answer, [evidence()]) == (True, [])


def test_abstention_threshold() -> None:
    assert should_abstain([evidence(0.01)], minimum_evidence=1, minimum_score=0.02)
