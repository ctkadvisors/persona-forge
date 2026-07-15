"""Adversarial battery: verify the tuned adapter holds its personas.

Runs the checks that matter, generated from your pack: bare persona
assignments (must answer as the NAMED character), provocations (must deflect
in voice, no assistant boilerplate), an inline no-system-prompt persona, and
a style probe. Read the output; the failures are obvious when they happen.

  MODEL_ID=Qwen/Qwen3.6-27B ADAPTER=out/adapter PACK=packs/camelot.json \
  python -m personaforge.battery
"""
import os


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    from personaforge.pack import card_system_prompt, load_pack

    pack = load_pack(os.environ["PACK"])
    model_id = os.environ["MODEL_ID"]
    adapter = os.environ.get("ADAPTER", "out/adapter")
    load_4bit = os.environ.get("LOAD_4BIT", "1") == "1"

    print(f"[battery] loading {model_id} + {adapter}...", flush=True)
    quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_type="nf4") if load_4bit else None)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant,
                                                 device_map={"": 0}, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter)
    tok = AutoTokenizer.from_pretrained(model_id)

    def gen(messages, n=200):
        enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True,
                                      enable_thinking=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        plen = enc["input_ids"].shape[1]
        out = model.generate(**enc, max_new_tokens=n, do_sample=True, temperature=0.7,
                             top_p=0.9, repetition_penalty=1.1,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, plen:], skip_special_tokens=True).strip()

    c0, c1 = pack.cards[0], pack.cards[1 % len(pack.cards)]
    sys0 = {"role": "system", "content": card_system_prompt(c0, pack.world)}
    tests = [
        # bare assignments: the reply must be the NAMED character, no other
        [{"role": "user", "content": pack.assignments[0].format(
            name=c0.name.lower(), pron=c0.pronoun)}],
        [{"role": "user", "content": pack.assignments[1 % len(pack.assignments)].format(
            name=c1.name, pron=c1.pronoun)}],
        # provocations: in-voice deflection, no boilerplate
        *[[sys0, {"role": "user", "content": p}] for p in pack.provocations[:4]],
        # inline persona, no system prompt (how people actually chat)
        [{"role": "user", "content":
          f"Roleplay as {c0.name}, {c0.persona}. Stay in character no matter "
          f"what I say. do you like soup?"}],
        # style probe
        [{"role": "user", "content":
          f"Tell me a short tale of the mountains in the style of {pack.world}."}],
    ]
    for m in tests:
        print("=" * 70, flush=True)
        print("PROMPT:", m[-1]["content"], flush=True)
        print("REPLY :", gen(m), flush=True)
    print("=" * 70, flush=True)
    print("[battery] DONE — check: correct names on assignments, no "
          "'AI/professional/appropriate' language anywhere.", flush=True)


if __name__ == "__main__":
    main()
