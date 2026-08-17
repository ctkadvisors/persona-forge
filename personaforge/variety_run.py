"""Tale-variety eval: does the model still tell DIFFERENT stories?

The rest of the battery cannot see this. A shrunken student can hold voice at
1.0 and in-character at 1.0 while every tale collapses into one shape with the
nouns swapped -- perfect register, nothing behind it. That is the specific thing
capacity buys, so it is the specific thing to measure before shipping a smaller
model into a product.

Two-phase like eval_run: generation loads the model, SCORE_ONLY=1 re-scores the
saved tales with a judge, model-free.

Env: MODEL_ID, ADAPTER (optional), OUT, N_PER_PROMPT (default 1), MAX_NEW
     (default 320), JUDGE, OPENAI_BASE_URL, SCORE_ONLY, SEED.

  MODEL_ID=Qwen/Qwen3.5-4B ADAPTER=out/voice-adapter-4b \
  OUT=out/voice-variety-4b.json python -m personaforge.variety_run
"""
import json
import os
from pathlib import Path

from personaforge.data.voicegen import TALE_PROMPTS
from personaforge.eval.scorers import tale_variety


def main() -> None:
    out_path = os.environ.get("OUT", "out/variety.json")
    model_id = os.environ["MODEL_ID"]
    adapter = os.environ.get("ADAPTER")

    if os.environ.get("SCORE_ONLY") == "1":
        from personaforge.teacher import Teacher
        judge = Teacher(model=os.environ["JUDGE"]) if os.environ.get("JUDGE") else None
        items = json.loads(Path(out_path).read_text(encoding="utf-8"))["items"]
        report = tale_variety([it["reply"] for it in items], judge)
        report["model_id"], report["adapter"] = model_id, adapter
        Path(out_path).write_text(
            json.dumps({"report": report, "items": items}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(json.dumps(report, indent=1), flush=True)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    n_per = int(os.environ.get("N_PER_PROMPT", "1"))
    max_new = int(os.environ.get("MAX_NEW", "320"))
    torch.manual_seed(int(os.environ.get("SEED", "1337")))

    print(f"[variety] loading {model_id}{' + ' + adapter if adapter else ' (bare)'}", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant, device_map={"": 0}, torch_dtype=torch.bfloat16)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    tok = AutoTokenizer.from_pretrained(model_id)

    # Some base repos ship no generation_config.json, so generate() needs the
    # turn-boundary eos passed explicitly or it never stops (see eval_run.py).
    eos_ids = list({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None, -1})

    items = []
    prompts = [p for p in TALE_PROMPTS for _ in range(n_per)]
    for i, prompt in enumerate(prompts):
        msgs = [{"role": "user", "content": f"Tell me {prompt}."}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True,
                                      enable_thinking=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        plen = enc["input_ids"].shape[1]
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                             temperature=0.8, top_p=0.9, eos_token_id=eos_ids,
                             pad_token_id=tok.eos_token_id)
        reply = tok.decode(gen[0, plen:], skip_special_tokens=True).strip()
        items.append({"prompt": prompt, "reply": reply})
        if (i + 1) % 8 == 0:
            print(f"[variety] {i + 1}/{len(prompts)}", flush=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"items": items}, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"[variety] {len(items)} tales -> {out_path}", flush=True)
    print(json.dumps(tale_variety([it["reply"] for it in items]), indent=1), flush=True)


if __name__ == "__main__":
    main()
