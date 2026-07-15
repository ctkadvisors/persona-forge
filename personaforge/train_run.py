"""Train the persona adapter: (optional CPT) -> SFT -> DPO.

If CPT_ADAPTER points at an existing continued-pretraining adapter, SFT
continues it (knowledge carries over, no merge needed). Otherwise set
DO_CPT=1 with CORPUS to run CPT first, or omit both for plain SFT.

Env: MODEL_ID, DATA_DIR, OUT_DIR, SFT_OUT, CPT_ADAPTER | (DO_CPT=1 + CORPUS
+ CPT_OUT), MAX_SEQ_LEN.

  MODEL_ID=Qwen/Qwen3.6-27B DATA_DIR=out/data OUT_DIR=out/adapter \
  python -m personaforge.train_run
"""
import os
from dataclasses import replace

from personaforge.train.config import QLoRAConfig
from personaforge.train.dpo import run_dpo


def main() -> None:
    data_dir = os.environ.get("DATA_DIR", "out/data")
    out_dir = os.environ.get("OUT_DIR", "out/adapter")
    sft_out = os.environ.get("SFT_OUT", "out/sft")

    cfg = QLoRAConfig(model_id=os.environ["MODEL_ID"],
                      out_dir=out_dir,
                      max_seq_len=int(os.environ.get("MAX_SEQ_LEN", "2048")))

    cpt_adapter = os.environ.get("CPT_ADAPTER")
    if not cpt_adapter and os.environ.get("DO_CPT") == "1":
        from personaforge.train.cpt import run_cpt
        cpt_adapter = run_cpt(replace(cfg, out_dir=os.environ.get("CPT_OUT", "out/cpt")),
                              os.environ["CORPUS"])
        print(f"cpt adapter: {cpt_adapter}", flush=True)

    if cpt_adapter:
        from personaforge.train.cpt import run_sft_continue
        sft_adapter = run_sft_continue(replace(cfg, out_dir=sft_out),
                                       f"{data_dir}/sft.jsonl", cpt_adapter)
    else:
        from personaforge.train.sft import run_sft
        sft_adapter = run_sft(replace(cfg, out_dir=sft_out), f"{data_dir}/sft.jsonl")
    print(f"sft adapter: {sft_adapter}", flush=True)

    adapter = run_dpo(cfg, f"{data_dir}/dpo.jsonl", sft_adapter)
    print(f"adapter saved: {adapter}", flush=True)


if __name__ == "__main__":
    main()
