import random

from personaforge.rag.chunk import chunk_text
from personaforge.data.synth import gen_grounded_qa, make_judge
from personaforge.data.blend import blend_sft, write_datasets


def build_blended_datasets(corpus_txt, teacher, pack, out_dir, n_chunks=200,
                           chat_n=800, dpo_n=500, seed=1):
    """The base training blend: corpus-grounded QA (teacher-generated,
    judge-filtered) + general chat (Dolly) + pack exemplar dialogues +
    preference pairs (UltraFeedback + a handful of in-voice pairs from the
    pack). Roleplay depth comes later from build_data's synthetic stages."""
    from personaforge.data import hf_load, baseline
    from personaforge.pack import exemplar_rows

    chunks = [c["text"] for c in chunk_text(corpus_txt, target_chars=900)]
    random.Random(seed).shuffle(chunks)
    qa = gen_grounded_qa(chunks[:n_chunks], teacher, make_judge(teacher), per_passage=1)

    dolly = hf_load.sample_split("databricks/databricks-dolly-15k", "train", chat_n, seed=seed)
    chat = baseline.to_sft_rows(dolly, {"user": "instruction", "assistant": "response"},
                                source="dolly", kind="chat")
    roleplay = exemplar_rows(pack)
    sft = blend_sft({"grounded_qa": qa, "chat": chat, "roleplay": roleplay},
                    caps={"grounded_qa": 100000, "chat": chat_n, "roleplay": 100000}, seed=seed)

    uf = hf_load.sample_split("HuggingFaceH4/ultrafeedback_binarized", "train_prefs", dpo_n, seed=seed)
    dpo = hf_load.ultrafeedback_to_dpo(uf, source="ultrafeedback")
    return write_datasets(sft, dpo, out_dir)


def run(cfg, datasets):
    from personaforge.train.sft import run_sft
    from personaforge.train.dpo import run_dpo
    adapter = run_sft(cfg, datasets["sft"])
    if datasets.get("dpo"):
        adapter = run_dpo(cfg, datasets["dpo"], adapter)
    return adapter


def run_with_cpt(cfg, corpus_path, datasets, cpt_out, sft_out):
    """Knowledge-first recipe: continued pretraining (CPT) on the raw corpus
    -> SFT continues that adapter -> DPO. Chains adapters (no fp16 merge
    needed between stages). cfg.out_dir is used for the final DPO adapter."""
    from dataclasses import replace
    from personaforge.train.cpt import run_cpt, run_sft_continue
    from personaforge.train.dpo import run_dpo

    cpt_adapter = run_cpt(replace(cfg, out_dir=cpt_out), corpus_path)
    sft_adapter = run_sft_continue(replace(cfg, out_dir=sft_out), datasets["sft"], cpt_adapter)
    if datasets.get("dpo"):
        return run_dpo(cfg, datasets["dpo"], sft_adapter)
    return sft_adapter
