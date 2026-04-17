"""Modal app: training function + inference function."""

from pathlib import Path

import modal

from training.config import TrainingConfig

app = modal.App("formrecap-lora")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git",
        "trl==0.9.6",
        "peft==0.12.0",
        "transformers==4.44.0",
        "datasets==2.21.0",
        "bitsandbytes==0.43.0",
        "accelerate==0.33.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

volume = modal.Volume.from_name("formrecap-lora", create_if_missing=True)
VOLUME_PATH = "/vol"


@app.function(
    image=image,
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
)
def train(run_id: str, train_file: str, val_file: str, config: dict | None = None) -> dict:
    """Fine-tune base model with LoRA. train_file/val_file are paths inside the volume."""
    import json
    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    cfg = TrainingConfig(**(config or {}))
    vol = Path(VOLUME_PATH)
    run_dir = vol / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load base in 4-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # 2. Attach LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        use_dora=cfg.lora.use_dora,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    # 3. Load datasets
    def format_prompts(examples):
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in examples["messages"]]
        return {"text": texts}

    ds_train = load_dataset("json", data_files=str(vol / train_file), split="train").map(format_prompts, batched=True)
    ds_val = load_dataset("json", data_files=str(vol / val_file), split="train").map(format_prompts, batched=True)

    # 4. Train
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=cfg.logging_steps,
        optim="adamw_8bit",
        weight_decay=0.0,
        seed=cfg.seed,
        eval_strategy=cfg.evaluation_strategy,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.epochs,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=False,
        args=training_args,
    )

    train_result = trainer.train()

    # 5. Save adapter + manifest
    model.save_pretrained(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "adapter"))

    # Identify best checkpoint by val loss
    log_history = trainer.state.log_history
    eval_losses = [(e.get("epoch"), e.get("eval_loss")) for e in log_history if "eval_loss" in e]
    best_epoch, best_loss = min(eval_losses, key=lambda x: x[1]) if eval_losses else (None, None)

    manifest = {
        "run_id": run_id,
        "train_file": train_file,
        "val_file": val_file,
        "config": {
            **cfg.__dict__,
            "lora": cfg.lora.__dict__,
        },
        "effective_batch_size": cfg.effective_batch_size,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "final_train_loss": train_result.training_loss,
        "eval_history": eval_losses,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    volume.commit()
    return manifest


@app.local_entrypoint()
def run_train(
    run_id: str = "baseline",
    train_file: str = "data/train.jsonl",
    val_file: str = "data/val.jsonl",
):
    manifest = train.remote(run_id=run_id, train_file=train_file, val_file=val_file)
    print(manifest)
