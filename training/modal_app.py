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
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    timeout=7200,
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

    raw_config = config or {}
    lora_overrides = raw_config.pop("lora", None)
    cfg = TrainingConfig(**raw_config)
    if lora_overrides:
        for k, v in lora_overrides.items():
            setattr(cfg.lora, k, v)
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
    def _adapt_messages(messages: list[dict]) -> list[dict]:
        """Merge system message into user message for models that don't support system role."""
        if not any(m["role"] == "system" for m in messages):
            return messages
        # Test if tokenizer supports system role
        try:
            tokenizer.apply_chat_template([{"role": "system", "content": "test"}], tokenize=False)
            return messages  # System role supported
        except Exception:
            pass
        # Merge system into first user message
        system = next(m["content"] for m in messages if m["role"] == "system")
        adapted = []
        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "user" and not adapted:
                adapted.append({"role": "user", "content": f"{system}\n\n{m['content']}"})
            else:
                adapted.append(m)
        return adapted

    def format_prompts(examples):
        texts = [
            tokenizer.apply_chat_template(
                _adapt_messages(m), tokenize=False, add_generation_prompt=False
            )
            for m in examples["messages"]
        ]
        return {"text": texts}

    ds_train = (
        load_dataset("json", data_files=str(vol / train_file), split="train")
        .map(format_prompts, batched=True)
        .remove_columns("messages")
    )
    ds_val = (
        load_dataset("json", data_files=str(vol / val_file), split="train")
        .map(format_prompts, batched=True)
        .remove_columns("messages")
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


@app.function(
    image=image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
)
def merge_adapter(run_id: str, base_model: str) -> str:
    """Merge a LoRA adapter into the base model for vLLM-compatible serving."""
    import os

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN", "")
    vol = Path(VOLUME_PATH)
    adapter_path = vol / "runs" / run_id / "adapter"
    merged_path = vol / "runs" / run_id / "merged"

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    base = AutoModelForCausalLM.from_pretrained(
        base_model, token=hf_token, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = model.merge_and_unload()

    merged.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))

    volume.commit()
    return f"Merged adapter saved to {merged_path}"


@app.local_entrypoint()
def run_merge(run_id: str = "baseline-3b", base_model: str = "meta-llama/Llama-3.2-3B-Instruct"):
    result = merge_adapter.remote(run_id=run_id, base_model=base_model)
    print(result)


@app.local_entrypoint()
def run_train(
    run_id: str = "baseline",
    train_file: str = "data/train.jsonl",
    val_file: str = "data/val.jsonl",
    base_model: str = "",
    lora_r: int = 0,
    lora_targets: str = "",
    no_dora: bool = False,
):
    config = {}
    if base_model:
        config["base_model"] = base_model
    lora_overrides = {}
    if lora_r > 0:
        lora_overrides["r"] = lora_r
        lora_overrides["alpha"] = lora_r * 2
    if lora_targets:
        lora_overrides["target_modules"] = lora_targets.split(",")
    if no_dora:
        lora_overrides["use_dora"] = False
    if lora_overrides:
        config["lora"] = lora_overrides
    manifest = train.remote(
        run_id=run_id,
        train_file=train_file,
        val_file=val_file,
        config=config or None,
    )
    print(manifest)


@app.cls(
    image=image,
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    timeout=600,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("huggingface")],
)
class Predictor:
    """Keeps model loaded in GPU memory across calls. First call cold-starts (~40s),
    subsequent calls are ~1-2s inference only."""

    run_id: str = modal.parameter(default="baseline-3b")
    base_model: str = modal.parameter(default="meta-llama/Llama-3.2-3B-Instruct")

    @modal.enter()
    def load_model(self):
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        hf_token = os.environ.get("HF_TOKEN", "")

        # Zero-shot mode: run_id="none" skips adapter loading
        if self.run_id == "none":
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, token=hf_token)
        else:
            vol = Path(VOLUME_PATH)
            adapter_path = vol / "runs" / self.run_id / "adapter"
            self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            token=hf_token,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        if self.run_id == "none":
            self.model = base
        else:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base, str(adapter_path))
        self.model.eval()

    @modal.method()
    def predict_with_logprobs(
        self,
        messages: list[dict],
        max_new_tokens: int = 128,
    ) -> dict:
        import torch

        # Adapt messages for models without system role support (Gemma, Mistral)
        adapted = messages
        if any(m["role"] == "system" for m in messages):
            try:
                self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": "test"}], tokenize=False
                )
            except Exception:
                system = next(m["content"] for m in messages if m["role"] == "system")
                adapted = []
                for m in messages:
                    if m["role"] == "system":
                        continue
                    if m["role"] == "user" and not adapted:
                        adapted.append({"role": "user", "content": f"{system}\n\n{m['content']}"})
                    else:
                        adapted.append(m)

        inputs = self.tokenizer.apply_chat_template(
            adapted, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output.sequences[0][inputs.shape[1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        first_scores = output.scores[0][0]
        first_probs = torch.softmax(first_scores, dim=-1)
        first_logprobs = torch.log_softmax(first_scores, dim=-1)

        topk = 10
        topk_vals, topk_ids = torch.topk(first_logprobs, topk)
        candidates = [
            {
                "token": self.tokenizer.decode([int(tid)]),
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
                "token": self.tokenizer.decode([emitted_first_token_id]),
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
    predictor = Predictor(run_id=run_id)
    result = predictor.predict_with_logprobs.remote(messages=messages)
    print(json.dumps(result, indent=2, default=str))
