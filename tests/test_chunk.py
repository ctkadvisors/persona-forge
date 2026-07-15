from personaforge.rag.chunk import chunk_text


def test_chunks_cover_text_and_respect_size():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    chunks = chunk_text(text, target_chars=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 700 for c in chunks)  # target + one paragraph slack
    assert [c["idx"] for c in chunks] == list(range(len(chunks)))
    assert "Paragraph 0" in chunks[0]["text"]


def test_small_text_single_chunk():
    assert len(chunk_text("short", target_chars=500)) == 1
