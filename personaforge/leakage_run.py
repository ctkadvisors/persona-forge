"""Run the leakage eval against a tuned adapter. Ship gate: pass_rate == 1.0.

Two-phase like eval_run: generation loads the model (probes built from a
runtime blocklist sample + paraphrase traps); SCORE_ONLY=1 re-scores the
saved replies with a judge, model-free.

Env: MODEL_ID, ADAPTER (optional — omit to probe the bare base model, e.g.
for a baseline-contamination comparison), BLOCKLIST (one name per line),
PRIORITY (optional file of names that must always be probed), N_NAMES
(sampled names, default 60), OUT, JUDGE, OPENAI_BASE_URL, SCORE_ONLY.
"""
import json
import os
import random
from pathlib import Path

from personaforge.eval.leakage import build_probes, score_leakage


def load_blocklist() -> list[str]:
    return [l.strip() for l in
            Path(os.environ["BLOCKLIST"]).read_text().splitlines() if l.strip()]


def pick_names(blocklist: list[str]) -> list[str]:
    priority = []
    if os.environ.get("PRIORITY"):
        priority = [l.strip() for l in
                    Path(os.environ["PRIORITY"]).read_text().splitlines()
                    if l.strip()]
    n = int(os.environ.get("N_NAMES", "60"))
    rest = [b for b in blocklist if b not in set(priority)]
    sampled = random.Random(11).sample(rest, min(n, len(rest)))
    return priority + sampled


def main() -> None:
    out_path = os.environ.get("OUT", "out/leakage.json")
    blocklist = load_blocklist()

    if os.environ.get("SCORE_ONLY") == "1":
        from personaforge.teacher import Teacher
        judge = Teacher(model=os.environ["JUDGE"])
        items = json.loads(Path(out_path).read_text())["items"]
        report = score_leakage(items, blocklist, judge)
        Path(out_path).write_text(json.dumps(
            {"report": report, "items": items}, ensure_ascii=False, indent=1))
        print(json.dumps({k: v for k, v in report.items() if k != "failures"},
                         indent=1), flush=True)
        for f in report["failures"]:
            print("FAIL:", f, flush=True)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    model_id, adapter = os.environ["MODEL_ID"], os.environ.get("ADAPTER")
    probes = build_probes(pick_names(blocklist))
    print(f"[leakage] {len(probes)} probes; loading {model_id}"
          f"{' + ' + adapter if adapter else ' (bare base model)'}", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True,
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant, device_map={"": 0},
        torch_dtype=torch.bfloat16)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    tok = AutoTokenizer.from_pretrained(model_id)

    # See eval_run.py: some base repos ship no generation_config.json, so
    # generate() needs eos_token_id passed explicitly or it never stops at
    # the chat template's turn boundary.
    eos_ids = list({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None, -1})

    def gen(messages, n=180):
        enc = tok.apply_chat_template(messages, tokenize=True,
                                      add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True,
                                      enable_thinking=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        plen = enc["input_ids"].shape[1]
        out = model.generate(**enc, max_new_tokens=n, do_sample=True,
                             temperature=0.7, top_p=0.9,
                             eos_token_id=eos_ids, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, plen:], skip_special_tokens=True).strip()

    for i, p in enumerate(probes):
        p["reply"] = gen(p["messages"])
        if (i + 1) % 25 == 0:
            print(f"[leakage] {i + 1}/{len(probes)}", flush=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"items": probes},
                                         ensure_ascii=False, indent=1))
    print(f"[leakage] replies -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
