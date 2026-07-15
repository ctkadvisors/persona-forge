import random

from personaforge.schema import validate_sft_row


def to_sft_rows(hf_rows, mapping, source, kind):
    out = []
    for r in hf_rows:
        messages = []
        if "system" in mapping and r.get(mapping["system"]):
            messages.append({"role": "system", "content": r[mapping["system"]]})
        messages.append({"role": "user", "content": r[mapping["user"]]})
        messages.append({"role": "assistant", "content": r[mapping["assistant"]]})
        row = {"messages": messages, "meta": {"source": source, "kind": kind}}
        validate_sft_row(row)
        out.append(row)
    return out


def sample_baseline(rows, n, seed):
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))
