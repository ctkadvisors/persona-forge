from personaforge.schema import read_jsonl


def render_default_prompt(messages) -> str:
    # Fallback renderer used when no HF tokenizer chat template is available (tests).
    return "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages) + "\n<|assistant|>\n"


def load_sft_dataset(path, tokenizer):
    from datasets import Dataset
    rows = read_jsonl(path)
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    return Dataset.from_dict({"text": texts})


def load_dpo_dataset(path, render_prompt=render_default_prompt, render_default_prompt_import=False):
    from datasets import Dataset
    rows = read_jsonl(path)
    return Dataset.from_dict({
        "prompt": [render_prompt(r["prompt"]) for r in rows],
        "chosen": [r["chosen"] for r in rows],
        "rejected": [r["rejected"] for r in rows],
    })
