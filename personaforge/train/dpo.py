def run_dpo(cfg, dpo_path: str, sft_adapter: str) -> str:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    from trl import DPOTrainer, DPOConfig
    from personaforge.train.chat_data import load_dpo_dataset

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_type="nf4") if cfg.load_in_4bit else None)
    # Pin to the single GPU (GB10 unified memory); 4-bit optional (MoE + bnb crashes).
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, quantization_config=quant, device_map={"": 0}, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, sft_adapter, is_trainable=True)

    def render(messages):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ds = load_dpo_dataset(dpo_path, render_prompt=render)
    args = DPOConfig(output_dir=cfg.out_dir, per_device_train_batch_size=cfg.batch_size,
                     gradient_accumulation_steps=cfg.grad_accum, learning_rate=5e-6,
                     num_train_epochs=cfg.epochs, bf16=True, logging_steps=10,
                     save_strategy="epoch", seed=cfg.seed, beta=0.1)
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(cfg.out_dir)
    return cfg.out_dir
