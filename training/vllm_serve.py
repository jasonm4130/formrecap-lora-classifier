"""vLLM-based inference server on Modal.

Serves Gemma 2B with optional LoRA adapter. Supports zero-shot (base model)
and full-lora inference on the same container via the OpenAI-compatible API.
"""

import modal

from training.config import VOLUME_PATH

app = modal.App("formrecap-lora-vllm")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .pip_install("vllm>=0.8,<1.0", "hf_transfer==0.1.9")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

volume = modal.Volume.from_name("formrecap-lora", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    scaledown_window=300,
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("vllm-api-key")],
)
@modal.concurrent(max_inputs=50)
@modal.web_server(port=8000, startup_timeout=600)
def serve_gemma_2b():
    import os
    import subprocess

    volume.reload()
    api_key = os.environ.get("VLLM_API_KEY", "")
    cmd = [
        "vllm",
        "serve",
        "google/gemma-2b-it",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        "512",
        "--enforce-eager",
        "--dtype",
        "bfloat16",
        "--enable-lora",
        "--max-lora-rank",
        "64",
        "--lora-modules",
        f"gemma-2b-cf={VOLUME_PATH}/runs/gemma-2b-cf/adapter",
        "--lora-modules",
        f"gemma-2b-nodora={VOLUME_PATH}/runs/gemma-2b-nodora/adapter",
    ]
    if api_key:
        cmd.extend(["--api-key", api_key])
    subprocess.Popen(" ".join(cmd), shell=True)
