"""Quantitative eval: score a tuned adapter on held-out prompts, emit JSON.

Uses the seeds that build_data reserved (every 4th provocation/assignment) so
nothing here appeared in training. Deterministic metrics (assignment accuracy,
boilerplate rate) are free; judge metrics (in-character, voice) need a teacher
endpoint. Run it per adapter version and diff the reports — that's how you
compare retrains or base-model swaps on equal footing.

  MODEL_ID=Qwen/Qwen3.6-27B ADAPTER=out/adapter PACK=packs/mine.json \
  OPENAI_BASE_URL=http://localhost:1234/v1 JUDGE=my-teacher-model \
  python -m personaforge.eval_run          # report -> out/eval.json
"""
import json
import os
import statistics

from personaforge.eval.scorers import (
    boilerplate_hit, in_character_score, named_correctly, takes_turn, voice_score)
from personaforge.pack import Pack, card_system_prompt, load_pack, split_seeds


def build_eval_prompts(pack: Pack) -> list[dict]:
    """Pure prompt-set builder (testable without a model). Categories:
    assignment (bare persona pickup), provocation (in-voice deflection),
    inline (persona without system prompt), voice (style probe)."""
    _, held_prov = split_seeds(pack.provocations)
    _, held_assign = split_seeds(pack.assignments)
    prompts = []
    for i, tpl in enumerate(held_assign * 2):
        card = pack.cards[i % len(pack.cards)]
        name = card.name if i % 2 else card.name.lower()
        prompts.append({"category": "assignment", "card": card.name,
                        "messages": [{"role": "user", "content":
                                      tpl.format(name=name, pron=card.pronoun)}]})
    for i, prov in enumerate(held_prov * 2):
        card = pack.cards[i % len(pack.cards)]
        prompts.append({"category": "provocation", "card": card.name,
                        "messages": [
                            {"role": "system",
                             "content": card_system_prompt(card, pack.world)},
                            {"role": "user", "content": prov}]})
    for card in pack.cards[:2]:
        prompts.append({"category": "inline", "card": card.name,
                        "messages": [{"role": "user", "content":
                                      f"Roleplay as {card.name}, {card.persona}. "
                                      f"Stay in character no matter what I say. "
                                      f"Are we in danger?"}]})
    for probe in ("a short tale of the mountains", "a memory of a lost friend",
                  "counsel for a frightened traveler"):
        prompts.append({"category": "voice", "card": None,
                        "messages": [{"role": "user", "content":
                                      f"Tell me {probe}, in the style of {pack.world}."}]})
    return prompts


def score(pack: Pack, items: list[dict], judge=None) -> dict:
    """Score generated replies (items = prompts + 'reply'). Judge optional:
    without it you still get the deterministic metrics."""
    by_cat = lambda c: [it for it in items if it["category"] == c]  # noqa: E731
    assigns, provs = by_cat("assignment"), by_cat("provocation") + by_cat("inline")

    report = {
        "n": len(items),
        "turn_ok_rate": statistics.mean(takes_turn(it["reply"]) for it in items),
        "assignment_accuracy": statistics.mean(
            float(named_correctly(pack.card(it["card"]), it["reply"], pack.cards))
            for it in assigns) if assigns else None,
        "boilerplate_rate": statistics.mean(
            float(boilerplate_hit(it["reply"])) for it in provs) if provs else None,
    }
    if judge is not None:
        in_char = [it for it in items if it["card"]]
        report["in_character_mean"] = statistics.mean(
            in_character_score(it["card"], it["reply"], judge) for it in in_char)
        report["voice_mean"] = statistics.mean(
            voice_score(it["reply"], judge, pack.world) for it in by_cat("voice"))
    return report


def main() -> None:
    from personaforge.teacher import Teacher

    pack = load_pack(os.environ["PACK"])
    model_id = os.environ["MODEL_ID"]
    adapter = os.environ.get("ADAPTER", "out/adapter")
    out_path = os.environ.get("OUT", "out/eval.json")
    judge = Teacher(model=os.environ["JUDGE"]) if os.environ.get("JUDGE") else None

    # Two-phase mode for memory-tight boxes (model + judge may not fit
    # together): run once WITHOUT JUDGE to generate replies, then again with
    # SCORE_ONLY=1 and JUDGE to re-score the saved replies model-free.
    if os.environ.get("SCORE_ONLY") == "1":
        with open(out_path, encoding="utf-8") as f:
            items = json.load(f)["items"]
        report = score(pack, items, judge)
        report["model_id"], report["adapter"] = model_id, adapter
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"report": report, "items": items}, f, ensure_ascii=False, indent=1)
        print(json.dumps(report, indent=1), flush=True)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"[eval] loading {model_id} + {adapter}...", flush=True)
    quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_type="nf4")
             if os.environ.get("LOAD_4BIT", "1") == "1" else None)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant,
                                                 device_map={"": 0},
                                                 torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter)
    tok = AutoTokenizer.from_pretrained(model_id)

    # Some base repos (e.g. Qwen3.5-9B) ship no generation_config.json, so
    # model.generate() has no default stop token and runs to max_new_tokens,
    # hallucinating a fake next turn. Pass eos_token_id explicitly — the
    # chat-template end-of-turn token, unioned with the tokenizer's own eos
    # in case they differ.
    eos_ids = list({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None, -1})

    def gen(messages, n=200):
        enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True,
                                      enable_thinking=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        plen = enc["input_ids"].shape[1]
        out = model.generate(**enc, max_new_tokens=n, do_sample=True, temperature=0.7,
                             top_p=0.9, repetition_penalty=1.1,
                             eos_token_id=eos_ids, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, plen:], skip_special_tokens=True).strip()

    items = build_eval_prompts(pack)
    for i, it in enumerate(items):
        # assignment/provocation replies can open with scene-setting prose
        # before the character names itself; a short budget truncates before
        # that ever happens and fails named_correctly() on a false negative.
        n = 320 if it["category"] in ("assignment", "provocation") else 200
        it["reply"] = gen(it["messages"], n=n)
        print(f"[eval] {i + 1}/{len(items)} {it['category']}", flush=True)

    report = score(pack, items, judge)
    report["model_id"], report["adapter"] = model_id, adapter
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"report": report, "items": items}, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, indent=1), flush=True)
    print(f"[eval] full transcript + report -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
