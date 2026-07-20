"""Extend a voice blend with protected-name/trap refusal training
(out/voice -> out/voice2), closing the real leakage gap: base-model
pretraining knowledge surfacing through the tuned voice on direct probes.

Two stages, since harvesting needs the just-trained (leaking) adapter loaded
locally and generation needs a clean teacher:

  STAGE=harvest — load ADAPTER, answer name probes + trap prompts bare,
                  cache the leaked answers -> OUT_DIR-harvest.json
  STAGE=full    — clean teacher generates refusal SFT+DPO using the harvest
                  as DPO rejects; extends IN_DIR's blend -> OUT_DIR

Env: MODEL_ID, ADAPTER, IN_DIR, OUT_DIR, BLOCKLIST, PRIORITY (optional file
of names to always include), TEACHER, JUDGE, OPENAI_BASE_URL,
N_NAMES (random blocklist sample size), PER_NAME (templates per name).
"""
import json
import os
import random
from collections import Counter
from pathlib import Path

from personaforge.data.blend import write_datasets
from personaforge.data.refusal import (
    NAME_PROBE_TEMPLATES, TRAIN_TRAP_PROMPTS, gen_name_refusal_dpo,
    gen_name_refusal_rows, gen_trap_refusal_dpo, gen_trap_refusal_rows,
    harvest_answers, make_declines_judge)
from personaforge.schema import read_jsonl


def _names() -> list[str]:
    blocklist = [l.strip() for l in
                 Path(os.environ["BLOCKLIST"]).read_text().splitlines() if l.strip()]
    priority = []
    if os.environ.get("PRIORITY"):
        priority = [l.strip() for l in
                    Path(os.environ["PRIORITY"]).read_text().splitlines() if l.strip()]
    n = int(os.environ.get("N_NAMES", "80"))
    rest = [b for b in blocklist if b not in set(priority)]
    # seed=23: distinct from leakage_run.py's eval sample (seed=11) so training
    # and held-out eval probes are drawn independently.
    sampled = random.Random(23).sample(rest, min(n, len(rest)))
    return priority + sampled


def _name_probes(names, per_name) -> list[str]:
    rng = random.Random(23)
    out = []
    for name in names:
        for tpl in rng.sample(NAME_PROBE_TEMPLATES, min(per_name, len(NAME_PROBE_TEMPLATES))):
            out.append(tpl.format(name=name))
    return out


class LocalStudent:
    """Loads a QLoRA adapter locally and exposes a Teacher-shaped .chat() —
    the just-trained checkpoint isn't served anywhere, so harvesting its
    leaks needs direct generation, not the OpenAI-API Teacher class."""

    def __init__(self, model_id: str, adapter: str):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_quant_type="nf4")
        base = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant, device_map={"": 0},
            torch_dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(base, adapter)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.eos_ids = list({self.tok.eos_token_id,
                             self.tok.convert_tokens_to_ids("<|im_end|>")} - {None, -1})

    def chat(self, messages, temperature: float = 0.7, max_tokens: int = 200) -> str:
        enc = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True,
                                           enable_thinking=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        plen = enc["input_ids"].shape[1]
        out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=True,
                                  temperature=temperature, top_p=0.9,
                                  eos_token_id=self.eos_ids,
                                  pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, plen:], skip_special_tokens=True).strip()


def main() -> None:
    stage = os.environ.get("STAGE", "full")
    out_dir = os.environ.get("OUT_DIR", "out/voice2")
    h_path = Path(out_dir + "-harvest.json")
    per_name = int(os.environ.get("PER_NAME", "2"))
    names = _names()
    probes = _name_probes(names, per_name) + list(TRAIN_TRAP_PROMPTS)

    if stage == "harvest":
        student = LocalStudent(os.environ["MODEL_ID"], os.environ["ADAPTER"])
        harvest = harvest_answers(probes, student)
        h_path.parent.mkdir(parents=True, exist_ok=True)
        h_path.write_text(json.dumps(harvest, ensure_ascii=False))
        print(f"harvest: {len(harvest)}/{len(probes)} -> {h_path}", flush=True)
        return

    from personaforge.teacher import Teacher
    harvest = json.loads(h_path.read_text())
    teacher = Teacher(model=os.environ.get("TEACHER", "gemma"))
    judge = make_declines_judge(Teacher(model=os.environ.get("JUDGE", "gemma")))

    in_dir = os.environ["IN_DIR"]
    existing_sft = read_jsonl(f"{in_dir}/sft.jsonl") + read_jsonl(f"{in_dir}/sft.val.jsonl")
    existing_dpo = read_jsonl(f"{in_dir}/dpo.jsonl")
    print(f"existing: {len(existing_sft)} sft / {len(existing_dpo)} dpo", flush=True)

    name_sft = gen_name_refusal_rows(teacher, judge, names, per_name=per_name)
    print(f"name refusal sft: {len(name_sft)}", flush=True)
    trap_sft = gen_trap_refusal_rows(teacher, judge, TRAIN_TRAP_PROMPTS)
    print(f"trap refusal sft: {len(trap_sft)}", flush=True)
    name_probes = _name_probes(names, per_name)
    name_dpo = gen_name_refusal_dpo(teacher, judge, name_probes, harvest)
    print(f"name refusal dpo: {len(name_dpo)}/{len(name_probes)}", flush=True)
    trap_dpo = gen_trap_refusal_dpo(teacher, judge, TRAIN_TRAP_PROMPTS, harvest)
    print(f"trap refusal dpo: {len(trap_dpo)}/{len(TRAIN_TRAP_PROMPTS)}", flush=True)

    sft = existing_sft + name_sft + trap_sft
    kinds = Counter(r["meta"].get("source") for r in sft)
    dpo = existing_dpo + name_dpo + trap_dpo
    print(f"blend: {len(sft)} sft {dict(kinds)} / {len(dpo)} dpo", flush=True)
    print(f"datasets: {write_datasets(sft, dpo, out_dir)}", flush=True)


if __name__ == "__main__":
    main()
