"""Continued pretraining (CPT) on the raw corpus — the knowledge-injection
stage that goes BEFORE SFT. Trains next-token on plain text (no chat template)
so the model gains deep familiarity with the legendarium before it learns to
chat. Full recipe: CPT -> SFT -> DPO. QLoRA; GPU libs lazy-imported."""


def run_cpt(cfg, corpus_path: str, chunk_chars: int = 2000, init_adapter: str | None = None) -> str:
    import torch
    from pathlib import Path
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_type="nf4") if cfg.load_in_4bit else None)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, quantization_config=quant, device_map={"": 0}, torch_dtype=torch.bfloat16)

    text = Path(corpus_path).read_text(encoding="utf-8")
    chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]
    ds = Dataset.from_dict({"text": chunks})

    peft_cfg = None
    if init_adapter:
        # Continue from an earlier adapter (chain CPT stages without merging).
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        peft_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                              lora_dropout=cfg.lora_dropout, task_type="CAUSAL_LM",
                              target_modules="all-linear")

    args = SFTConfig(output_dir=cfg.out_dir, per_device_train_batch_size=cfg.batch_size,
                     gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.lr,
                     num_train_epochs=cfg.epochs, max_length=cfg.max_seq_len, bf16=True,
                     logging_steps=10, save_strategy="epoch", seed=cfg.seed, packing=True)
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         peft_config=peft_cfg, processing_class=tok)
    trainer.train()
    trainer.save_model(cfg.out_dir)
    return cfg.out_dir


def run_sft_continue(cfg, sft_path: str, init_adapter: str) -> str:
    """SFT that continues training an existing adapter (e.g. the CPT adapter),
    so CPT knowledge carries into the chat-tuning stage without a merge."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    from trl import SFTTrainer, SFTConfig
    from personaforge.train.chat_data import load_sft_dataset

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_type="nf4") if cfg.load_in_4bit else None)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, quantization_config=quant, device_map={"": 0}, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, init_adapter, is_trainable=True)
    ds = load_sft_dataset(sft_path, tok)
    args = SFTConfig(output_dir=cfg.out_dir, per_device_train_batch_size=cfg.batch_size,
                     gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.lr,
                     num_train_epochs=cfg.epochs, max_length=cfg.max_seq_len, bf16=True,
                     logging_steps=10, save_strategy="epoch", seed=cfg.seed)
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(cfg.out_dir)
    return cfg.out_dir
