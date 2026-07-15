"""Merge the trained adapter into the base model for quantization/serving.

Produces a standard bf16 HF checkpoint at OUT — feed that to llama.cpp's
convert_hf_to_gguf.py or `mlx_lm convert -q` (or serve it directly).

  MODEL_ID=Qwen/Qwen3.6-27B ADAPTER=out/adapter OUT=out/merged \
  python -m personaforge.merge_export

Gotchas learned the hard way (printed again at the end of a run):
- Exotic architectures may export a `model_type` your quantizer doesn't map
  (e.g. a `..._text` inner-config name). If conversion says "model type not
  supported", check for a close sibling and patch config.json in a shim copy.
- If the base has a thinking mode, check the exported chat template's DEFAULT
  before shipping — some inference stacks force-inject enable_thinking=True.
- If peft's merge complains about a quantization dispatcher (torchao et al.),
  uninstall the conflicting package in the training environment.
"""
import os
import time


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
    print("[merge] before quantizing, check: (1) config.json model_type is one "
          "your quantizer supports; (2) the chat template's thinking-mode "
          "DEFAULT is what you want shipped.", flush=True)


if __name__ == "__main__":
    main()
