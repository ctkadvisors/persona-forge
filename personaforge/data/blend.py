import json
import random
from pathlib import Path

from personaforge.schema import write_jsonl


def _key(row):
    return json.dumps(row["messages"], sort_keys=True, ensure_ascii=False)


def blend_sft(groups, caps, seed):
    rng = random.Random(seed)
    seen, out = set(), []
    for name, rows in groups.items():
        picked = rng.sample(rows, min(caps.get(name, len(rows)), len(rows)))
        for r in picked:
            k = _key(r)
            if k not in seen:
                seen.add(k)
                out.append(r)
    rng.shuffle(out)
    return out


def split_train_val(rows, val_frac, seed):
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_frac)
    return shuffled[n_val:], shuffled[:n_val]


def write_datasets(sft_rows, dpo_rows, out_dir):
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    tr, va = split_train_val(sft_rows, 0.05, seed=1) if sft_rows else ([], [])
    write_jsonl(tr or sft_rows, str(d / "sft.jsonl"))
    write_jsonl(va, str(d / "sft.val.jsonl"))
    write_jsonl(dpo_rows, str(d / "dpo.jsonl"))
    return {"sft": str(d / "sft.jsonl"), "sft_val": str(d / "sft.val.jsonl"),
            "dpo": str(d / "dpo.jsonl")}
