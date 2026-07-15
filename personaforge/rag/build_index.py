# /// script
# requires-python = ">=3.10"
# dependencies = ["sentence-transformers", "numpy"]
# ///
"""Build the local lore index: chunk a corpus, embed, save vectors + texts.

  uv run personaforge/rag/build_index.py <corpus.txt> <index_dir>

Writes <index_dir>/lore.npz (float32 vectors, no pickle) and lore.texts.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from personaforge.rag.chunk import chunk_text  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main(corpus: str, index_dir: str) -> None:
    corpus_path = Path(corpus)
    out_dir = Path(index_dir)
    chunks = chunk_text(corpus_path.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    print(f"{len(texts)} chunks from {corpus_path} ({corpus_path.stat().st_size // 1024} KiB)")

    model = SentenceTransformer(MODEL)
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True,
                           normalize_embeddings=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "lore.npz", vectors=vectors.astype(np.float32))
    (out_dir / "lore.texts.json").write_text(json.dumps(texts, ensure_ascii=False),
                                             encoding="utf-8")
    print(f"wrote {out_dir}/lore.npz + lore.texts.json (dim={vectors.shape[1]})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
