# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp", "sentence-transformers", "numpy"]
# ///
"""Lore-search MCP server: semantic search over your private corpus.

Stdio MCP server exposing `lore_search`. Register in an MCP host (LM Studio,
Claude Code, etc.) as:

  LORE_INDEX_DIR=/path/to/index LORE_DESCRIPTION="the collected works of X" \
  uv run <path>/mcp_server.py

Requires the index built by build_index.py. LORE_DESCRIPTION is spliced into
the tool description so the model knows what the corpus contains.
"""
import json
import os
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

INDEX_DIR = Path(os.environ.get("LORE_INDEX_DIR", "data/index"))
DESCRIPTION = os.environ.get("LORE_DESCRIPTION", "the source texts of this fictional world")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

mcp = FastMCP("loremaster")
_state: dict = {}


def _load():
    if not _state:
        data = np.load(INDEX_DIR / "lore.npz")  # plain float32 arrays; pickle stays disabled
        _state["vectors"] = data["vectors"]
        _state["texts"] = json.loads(
            (INDEX_DIR / "lore.texts.json").read_text(encoding="utf-8"))
        from sentence_transformers import SentenceTransformer
        _state["model"] = SentenceTransformer(MODEL)
    return _state


@mcp.tool(description=f"Search {DESCRIPTION} for passages relevant to a query. "
          "Use this to ground answers about the world's lore, characters, and "
          "events in the actual text instead of memory.")
def lore_search(query: str, top_k: int = 5) -> str:
    s = _load()
    q = s["model"].encode([query], normalize_embeddings=True)[0]
    scores = s["vectors"] @ q
    top_k = max(1, min(int(top_k), 12))
    order = np.argsort(-scores)[:top_k]
    results = [
        {"rank": i + 1, "score": round(float(scores[j]), 3), "passage": s["texts"][j]}
        for i, j in enumerate(order)
    ]
    return json.dumps(results, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
