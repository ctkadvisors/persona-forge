"""Merge the trained adapter into the base model for quantization/serving.

Produces a standard bf16 HF checkpoint at OUT — feed that to llama.cpp's
convert_hf_to_gguf.py or `mlx_lm convert -q` (or serve it directly).

  MODEL_ID=Qwen/Qwen3.6-27B ADAPTER=out/adapter OUT=out/merged \
  python -m personaforge.merge_export

Gotchas learned the hard way (printed again at the end of a run):
- Exotic architectures may export a `model_type` your quantizer doesn't map
  (e.g. a `..._text` inner-config name). If conversion says "model type not
  supported", check for a close sibling and patch config.json in a shim copy.
- If the base has a thinking mode, the export now hardcodes it OFF at the
  generation boundary (harden_chat_template, THINKING=keep opts out). This used
  to be a "remember to check it" note and that failed: the 4B shipped a template
  defaulting thinking ON, so the model burned its budget in reasoning_content
  and returned an EMPTY content field. Nothing crashes — the caller just gets
  nothing, which is why a load test alone does not catch it.
- If peft's merge complains about a quantization dispatcher (torchao et al.),
  uninstall the conflicting package in the training environment.
- If the base config has `mtp_num_hidden_layers` >= 1 (MTP/multi-token
  prediction — qwen3.5 family, DENSE MODELS INCLUDED), the merge drops the
  mtp.* tensors but convert_hf_to_gguf.py still counts them in block_count.
  llama.cpp then fails with "missing tensor blk.N..." for a block that does
  not exist. Convert with `--no-mtp`, or repair an already-broken GGUF in
  place with scripts/fix_gguf_mtp.py.
"""
import json
import os
import re
import time
from pathlib import Path

# The generation-boundary thinking toggle in the Qwen3.5-family chat template,
# whichever way round the size in question happens to phrase it.
_THINK_TOGGLE = re.compile(r"[ \t]*\{%-? if enable_thinking\b.*?\{%-? endif %\}\n?", re.S)
_THINK_OFF = "    {{- '<think>\\n\\n</think>\\n\\n' }}\n"


def harden_chat_template(out: str) -> str:
    """Hardcode thinking OFF at the generation boundary. Returns what it did.

    The stock Qwen3.5 template leaves thinking ON by default at some sizes (4B
    does; 2B defaults off but stays toggleable). Left alone the model spends its
    whole budget in reasoning_content and returns an EMPTY content field —
    nothing crashes, the caller just gets nothing back, which is worse. A toggle
    is not reliable either: some stacks force-inject enable_thinking.

    So rewrite the conditional to emit the empty <think></think> pair
    unconditionally, matching what shipped for the 9B. THINKING=keep opts out.
    """
    if os.environ.get("THINKING") == "keep":
        print("[merge] THINKING=keep — chat template left untouched", flush=True)
        return "kept"

    changed, suspect = [], []
    for name in ("chat_template.jinja", "tokenizer_config.json"):
        p = Path(out) / name
        if not p.exists():
            continue
        is_json = name.endswith(".json")
        raw = p.read_text(encoding="utf-8")
        s = json.loads(raw).get("chat_template") or "" if is_json else raw
        if not s or "enable_thinking" not in s:
            continue
        # NOTE: lambda replacement on purpose — re.sub processes backslash
        # escapes in a plain replacement string, which would write real newline
        # bytes where the template needs a literal two-character \n.
        new, n = _THINK_TOGGLE.subn(lambda _m: _THINK_OFF, s)
        if n != 1 or "enable_thinking" in new:
            suspect.append(f"{name} (matched {n})")
            continue
        if is_json:
            d = json.loads(raw)
            d["chat_template"] = new
            p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            p.write_text(new, encoding="utf-8")
        changed.append(name)

    if suspect:
        print(f"[merge] WARNING: could not harden {', '.join(suspect)} — "
              "CHECK THE TEMPLATE BY HAND before shipping", flush=True)
    if changed:
        print(f"[merge] chat template: thinking hardcoded OFF in {', '.join(changed)}",
              flush=True)
        return "hardened"
    return "no-toggle" if not suspect else "suspect"


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_id = os.environ["MODEL_ID"]
    adapter = os.environ.get("ADAPTER", "out/adapter")
    out = os.environ.get("OUT", "out/merged")

    t0 = time.time()
    print(f"[merge] loading {model_id} bf16 + {adapter}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                 device_map={"": 0}, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, adapter)
    print(f"[merge] merging ({time.time() - t0:.0f}s)...", flush=True)
    model = model.merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out)
    print(f"[merge] DONE -> {out}  total={time.time() - t0:.0f}s", flush=True)
    harden_chat_template(out)
    print("[merge] before quantizing, check config.json model_type is one your "
          "quantizer supports.", flush=True)
    mtp = getattr(model.config, "mtp_num_hidden_layers", 0) or 0
    if mtp:
        print(f"[merge] WARNING: base config has mtp_num_hidden_layers={mtp} "
              "but the merge dropped the mtp.* tensors — convert to GGUF with "
              "--no-mtp or the file will fail to load (missing tensor blk.N). "
              "scripts/fix_gguf_mtp.py repairs an already-converted file.",
              flush=True)


if __name__ == "__main__":
    main()
