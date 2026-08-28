from evidencebench.chunking import TextChunker, whitespace_token_count


def test_chunker_respects_limit_and_overlap() -> None:
    chunker = TextChunker(token_counter=whitespace_token_count, max_tokens=5, overlap_tokens=2)
    chunks = chunker.split("one two\n\nthree four\n\nfive six")
    assert [chunk.content for chunk in chunks] == [
        "one two\n\nthree four",
        "three four\n\nfive six",
    ]
    assert all(chunk.token_count <= 5 for chunk in chunks)


def test_chunker_splits_oversized_unit() -> None:
    chunker = TextChunker(token_counter=whitespace_token_count, max_tokens=3, overlap_tokens=0)
    chunks = chunker.split("one two three four five")
    assert [chunk.token_count for chunk in chunks] == [3, 2]
