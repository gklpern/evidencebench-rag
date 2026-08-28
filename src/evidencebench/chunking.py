import re
from collections.abc import Callable

from evidencebench.domain import ChunkDraft


class TextChunker:
    def __init__(
        self,
        *,
        token_counter: Callable[[str], int],
        max_tokens: int = 384,
        overlap_tokens: int = 48,
    ) -> None:
        if max_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("Require max_tokens > overlap_tokens >= 0")
        self.token_counter = token_counter
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def split(self, text: str) -> list[ChunkDraft]:
        normalized = text.replace("\r\n", "\n").strip()
        if not normalized:
            return []
        units = self._units(normalized)
        chunks: list[ChunkDraft] = []
        buffer: list[str] = []
        for unit in units:
            if buffer and self.token_counter("\n\n".join([*buffer, unit])) > self.max_tokens:
                chunks.append(self._draft(len(chunks), buffer))
                buffer = self._overlap(buffer)
            buffer.append(unit)
        if buffer:
            chunks.append(self._draft(len(chunks), buffer))
        return chunks

    def _units(self, text: str) -> list[str]:
        units = []
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if self.token_counter(paragraph) <= self.max_tokens:
                units.append(paragraph)
                continue
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            for sentence in sentences:
                if self.token_counter(sentence) <= self.max_tokens:
                    units.append(sentence)
                else:
                    units.extend(self._split_oversized(sentence))
        return units

    def _split_oversized(self, text: str) -> list[str]:
        words = text.split()
        parts: list[str] = []
        current: list[str] = []
        for word in words:
            if current and self.token_counter(" ".join([*current, word])) > self.max_tokens:
                parts.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            parts.append(" ".join(current))
        return parts

    def _overlap(self, units: list[str]) -> list[str]:
        overlap: list[str] = []
        for unit in reversed(units):
            candidate = [unit, *overlap]
            if self.token_counter("\n\n".join(candidate)) > self.overlap_tokens:
                break
            overlap = candidate
        return overlap

    def _draft(self, ordinal: int, units: list[str]) -> ChunkDraft:
        content = "\n\n".join(units)
        return ChunkDraft(ordinal=ordinal, content=content, token_count=self.token_counter(content))


def whitespace_token_count(text: str) -> int:
    return len(text.split())
