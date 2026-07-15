"""Load general-chat + preference baseline datasets from HuggingFace into our
schema. The `datasets.load_dataset` call is isolated in `_load_split` so tests
run on fixtures without network."""
import random

from personaforge.schema import validate_sft_row, validate_dpo_row


def _content(turn):
    """A conversation turn may be a plain string or a {role/from, content/value} dict."""
    if isinstance(turn, str):
        return turn
    return turn.get("content", turn.get("value", ""))


def conversations_to_sft(examples, conv_field, source, kind="chat"):
    """Map rows whose `conv_field` is an alternating [user, assistant, ...] list
    (of strings or dicts) into validated SFT chat rows."""
    out = []
    for ex in examples:
        turns = ex[conv_field]
        messages = []
        for i, turn in enumerate(turns):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": _content(turn)})
        if len(messages) < 2:
            continue
        row = {"messages": messages, "meta": {"source": source, "kind": kind}}
        validate_sft_row(row)
        out.append(row)
    return out


def ultrafeedback_to_dpo(examples, source):
    """Map UltraFeedback-style rows (prompt + chosen/rejected message lists or
    strings) into validated DPO rows."""
    out = []
    for ex in examples:
        prompt = ex["prompt"] if isinstance(ex["prompt"], str) else _content(ex["prompt"][-1])
        chosen = ex["chosen"][-1]["content"] if isinstance(ex["chosen"], list) else ex["chosen"]
        rejected = ex["rejected"][-1]["content"] if isinstance(ex["rejected"], list) else ex["rejected"]
        if not (prompt and chosen and rejected):
            continue
        row = {"prompt": [{"role": "user", "content": prompt}],
               "chosen": chosen, "rejected": rejected, "meta": {"source": source}}
        validate_dpo_row(row)
        out.append(row)
    return out


def _load_split(dataset_id, split, config=None):
    from datasets import load_dataset
    return list(load_dataset(dataset_id, config, split=split))


def sample_split(dataset_id, split, n, seed, config=None):
    rows = _load_split(dataset_id, split, config)
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))
