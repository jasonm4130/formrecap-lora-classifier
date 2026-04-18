"""Modal app: training function + inference function.

Uses plain HF stack (transformers + peft + trl + bitsandbytes) instead of Unsloth
to avoid dependency conflicts. For 884 training examples on an L4, the ~2x speedup
from Unsloth is not worth the integration headaches.
"""

from pathlib import Path

import modal

from training.config import TrainingConfig

app = modal.App("formrecap-lora")

# Plain HF stack — no Unsloth, no dependency hell.
# Pinned versions known to work together (April 2026).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.48.3",
        "peft==0.15.1",
        "trl==0.16.1",
        "datasets==3.5.0",
        "accelerate==1.5.0",
        "bitsandbytes==0.45.4",
        "huggingface_hub==0.29.3",
        "hf_transfer==0.1.9",
        "rich==14.0.0",
        "scipy>=1.13",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("training")
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
    import os

    import torch
    from datasets import load_dataset
    from peft import LoraConfig as PeftLoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    cfg = TrainingConfig(**(config or {}))
    vol = Path(VOLUME_PATH)
    run_dir = vol / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # HF token for gated model access (Llama 3.2 is a gated model)
    hf_token = os.environ.get("HF_TOKEN", "")

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load base model in 4-bit (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    model.config.use_cache = False

    # 3. LoRA config
    peft_config = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        use_dora=cfg.lora.use_dora,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Load datasets — format messages into text via chat template
    def format_prompts(examples):
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in examples["messages"]
        ]
        return {"text": texts}

    ds_train = load_dataset("json", data_files=str(vol / train_file), split="train").map(
        format_prompts, batched=True
    )
    ds_val = load_dataset("json", data_files=str(vol / val_file), split="train").map(
        format_prompts, batched=True
    )

    # 5. SFT training config
    sft_config = SFTConfig(
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
        max_length=cfg.max_length,
        packing=False,
        dataset_text_field="text",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 6. Create trainer with PEFT config
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        peft_config=peft_config,
        args=sft_config,
    )

    train_result = trainer.train()

    # 7. Save adapter + manifest
    trainer.save_model(str(run_dir / "adapter"))
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


@app.function(
    image=image,
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    timeout=600,
    min_containers=0,
    secrets=[modal.Secret.from_name("huggingface")],
)
def predict_with_logprobs(
    run_id: str,
    messages: list[dict],
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
    max_new_tokens: int = 128,
) -> dict:
    """Run inference with a fine-tuned adapter. Returns predicted text + first-token logprobs."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    vol = Path(VOLUME_PATH)
    adapter_path = vol / "runs" / run_id / "adapter"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

    # Load base model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA adapter on top
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output.sequences[0][inputs.shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # First-token logprobs
    first_scores = output.scores[0][0]  # (vocab_size,)
    first_probs = torch.softmax(first_scores, dim=-1)
    first_logprobs = torch.log_softmax(first_scores, dim=-1)

    # Top-K candidates for the first token
    topk = 10
    topk_vals, topk_ids = torch.topk(first_logprobs, topk)
    candidates = [
        {
            "token": tokenizer.decode([int(tid)]),
            "token_id": int(tid),
            "logprob": float(v),
            "prob": float(first_probs[tid]),
        }
        for tid, v in zip(topk_ids, topk_vals, strict=True)
    ]

    emitted_first_token_id = int(generated_ids[0])
    emitted_first_logprob = float(first_logprobs[emitted_first_token_id])

    return {
        "text": text,
        "first_token": {
            "id": emitted_first_token_id,
            "token": tokenizer.decode([emitted_first_token_id]),
            "logprob": emitted_first_logprob,
            "prob": float(first_probs[emitted_first_token_id]),
        },
        "top_candidates": candidates,
    }


@app.local_entrypoint()
def run_predict_smoke(run_id: str = "baseline-3b"):
    import json

    messages = [
        {
            "role": "system",
            "content": (
                "You analyse form interaction event sequences and classify the likely "
                "abandonment reason. Respond with a digit class code 1-6 on the first line, "
                "then a JSON object on the second line with class, reason, and confidence fields. "
                "Classes: 1=validation_error, 2=distraction, 3=comparison_shopping, "
                "4=accidental_exit, 5=bot, 6=committed_leave."
            ),
        },
        {
            "role": "user",
            "content": "Events: focus:email, input:email(x20), blur:email(invalid_format), exit",
        },
    ]
    result = predict_with_logprobs.remote(run_id=run_id, messages=messages)
    print(json.dumps(result, indent=2, default=str))
