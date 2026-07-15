from dataclasses import dataclass


@dataclass
class QLoRAConfig:
    model_id: str
    out_dir: str
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    batch_size: int = 1
    grad_accum: int = 16
    lr: float = 2e-4
    epochs: int = 1
    max_seq_len: int = 2048
    seed: int = 1337
