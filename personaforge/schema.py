import json
from typing import TypedDict

Message = TypedDict("Message", {"role": str, "content": str})

ROLES = {"system", "user", "assistant"}
KINDS = {"chat", "roleplay", "grounded_qa"}


def _check_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError(f"bad message: {m!r}")
        if m["role"] not in ROLES:
            raise ValueError(f"bad role: {m['role']!r}")
        if not isinstance(m["content"], str):
            raise ValueError("content must be str")


def validate_sft_row(row: dict) -> None:
    _check_messages(row.get("messages"))
    meta = row.get("meta", {})
    if meta.get("kind") not in KINDS:
        raise ValueError(f"bad kind: {meta.get('kind')!r}")


def validate_dpo_row(row: dict) -> None:
    _check_messages(row.get("prompt"))
    for k in ("chosen", "rejected"):
        if not isinstance(row.get(k), str) or not row[k]:
            raise ValueError(f"{k} must be a non-empty str")


def write_jsonl(rows: list[dict], path: str) -> int:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
