"""Central hyperparameter config. Single source of truth for all runs."""

from dataclasses import dataclass, field


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_dora: bool = True


@dataclass
class TrainingConfig:
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_length: int = 512
    load_in_4bit: bool = True  # QLoRA NF4
    learning_rate: float = 2e-4
    epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    logging_steps: int = 5
    lora: LoraConfig = field(default_factory=LoraConfig)

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps
