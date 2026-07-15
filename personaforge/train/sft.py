def run_sft(cfg, sft_path: str) -> str:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from personaforge.train.chat_data import load_sft_dataset

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 4-bit (bitsandbytes) is optional: MoE models (e.g. Qwen3.6-A3B) crash under
    # bnb-4bit on GB10/ARM, so load_in_4bit=False loads bf16 for plain LoRA.
    quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_type="nf4") if cfg.load_in_4bit else None)
    # Pin to the single GPU (GB10 unified memory); avoids accelerate offloading.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, quantization_config=quant, device_map={"": 0},
        torch_dtype=torch.bfloat16)
    peft_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                          lora_dropout=cfg.lora_dropout, task_type="CAUSAL_LM",
                          target_modules="all-linear")
    ds = load_sft_dataset(sft_path, tok)
    args = SFTConfig(output_dir=cfg.out_dir, per_device_train_batch_size=cfg.batch_size,
                     gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.lr,
                     num_train_epochs=cfg.epochs, max_length=cfg.max_seq_len,
                     bf16=True, logging_steps=10, save_strategy="epoch", seed=cfg.seed)
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         peft_config=peft_cfg, processing_class=tok)
    trainer.train()
    trainer.save_model(cfg.out_dir)
    return cfg.out_dir
