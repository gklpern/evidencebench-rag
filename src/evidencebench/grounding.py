import re

from evidencebench.domain import FusedCandidate, GroundedAnswer, Verdict


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def validate_citations(
    answer: GroundedAnswer, evidence: list[FusedCandidate]
) -> tuple[bool, list[str]]:
    by_chunk = {candidate.chunk_id: candidate for candidate in evidence}
    errors = []
    for citation in answer.citations:
        candidate = by_chunk.get(citation.chunk_id)
        if candidate is None:
            errors.append(f"{citation.citation_id}: unknown chunk")
            continue
        if normalize_text(citation.quote) not in normalize_text(candidate.content):
            errors.append(f"{citation.citation_id}: quote not found verbatim")
    if answer.verdict == Verdict.ANSWERED and not answer.citations:
        errors.append("answered response has no citations")
    if answer.verdict == Verdict.ABSTAINED and answer.answer.strip():
        errors.append("abstained response must not contain an answer")
    return not errors, errors


def should_abstain(
    evidence: list[FusedCandidate], *, minimum_evidence: int, minimum_score: float
) -> bool:
    qualifying = [candidate for candidate in evidence if candidate.score >= minimum_score]
    return len(qualifying) < minimum_evidence
