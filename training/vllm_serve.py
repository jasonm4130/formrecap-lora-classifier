"""vLLM-based inference server on Modal. Much faster than raw HuggingFace generate().

Supports zero-shot (base model) and LoRA adapter inference on the same server.
Logprobs come via the OpenAI-compatible API for free.
"""

import modal

app = modal.App("formrecap-lora-vllm")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .pip_install("vllm>=0.8,<1.0", "hf_transfer==0.1.9")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

volume = modal.Volume.from_name("formrecap-lora", create_if_missing=True)
VOLUME_PATH = "/vol"


def _build_vllm_cmd(
    base_model: str,
    lora_adapters: dict[str, str] | None = None,
    quantization: str | None = None,
) -> list[str]:
    """Build the vllm serve command."""
    cmd = [
        "vllm",
        "serve",
        base_model,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        "512",
        "--enforce-eager",
        "--dtype",
        "bfloat16",
    ]
    if quantization:
        cmd.extend(["--quantization", quantization, "--load-format", quantization])
    if lora_adapters:
        cmd.append("--enable-lora")
        cmd.append("--max-lora-rank")
        cmd.append("64")
        for name, path in lora_adapters.items():
            cmd.extend(["--lora-modules", f"{name}={path}"])
    return cmd


# --- Llama 3.2 3B ---
@app.function(
    image=vllm_image,
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    scaledown_window=300,
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface")],
)
@modal.concurrent(max_inputs=50)
@modal.web_server(port=8000, startup_timeout=600)
def serve_llama_3b():
    import subprocess

    cmd = _build_vllm_cmd(
        "meta-llama/Llama-3.2-3B-Instruct",
        lora_adapters={
            "baseline-3b": f"{VOLUME_PATH}/runs/baseline-3b/adapter",
        },
    )
    subprocess.Popen(" ".join(cmd), shell=True)


# --- Gemma 2B ---
@app.function(
    image=vllm_image,
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    scaledown_window=300,
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface")],
)
@modal.concurrent(max_inputs=50)
@modal.web_server(port=8000, startup_timeout=600)
def serve_gemma_2b():
    import subprocess

    volume.reload()
    cmd = _build_vllm_cmd(
        "google/gemma-2b-it",
        lora_adapters={
            "gemma-2b-cf": f"{VOLUME_PATH}/runs/gemma-2b-cf/adapter",
            "gemma-2b-nodora": f"{VOLUME_PATH}/runs/gemma-2b-nodora/adapter",
        },
    )
    subprocess.Popen(" ".join(cmd), shell=True)


# --- Mistral 7B ---
@app.function(
    image=vllm_image,
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    scaledown_window=300,
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface")],
)
@modal.concurrent(max_inputs=50)
@modal.web_server(port=8000, startup_timeout=600)
def serve_mistral_7b():
    import os
    import subprocess

    # Force volume reload and verify adapter exists
    volume.reload()
    adapter_path = f"{VOLUME_PATH}/runs/mistral-7b-cf/adapter"
    files = os.listdir(adapter_path)
    print(f"[vllm_serve] Adapter files at {adapter_path}: {files}")

    cmd = _build_vllm_cmd(
        "mistralai/Mistral-7B-Instruct-v0.2",
        lora_adapters={
            "mistral-7b-cf": f"{VOLUME_PATH}/runs/mistral-7b-cf/adapter",
        },
    )
    subprocess.Popen(" ".join(cmd), shell=True)
