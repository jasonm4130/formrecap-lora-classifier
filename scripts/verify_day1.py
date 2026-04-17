"""Day-1 verification. Run before any other implementation work."""

import os
import sys

import httpx
from dotenv import load_dotenv
from rich import print as rprint
from rich.table import Table

load_dotenv()

results = {}


def check(name: str, fn):
    try:
        fn()
        results[name] = ("PASS", "")
    except Exception as e:
        results[name] = ("FAIL", str(e))


# 1. HF license for Llama 3.2 3B
def check_hf():
    token = os.environ["HF_TOKEN"]
    r = httpx.get(
        "https://huggingface.co/api/models/meta-llama/Llama-3.2-3B-Instruct",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    if r.status_code == 200:
        body = r.json()
        if body.get("gated") and not body.get("gated_granted", False):
            # Some orgs return 200 even if not granted; hit the resolve endpoint
            r2 = httpx.head(
                "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/config.json",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
                follow_redirects=True,
            )
            assert r2.status_code == 200, f"License not accepted? HEAD returned {r2.status_code}"
    else:
        raise RuntimeError(f"Cannot access model: {r.status_code}")


# 2. Modal auth + GPU availability
def check_modal():
    import modal

    # dry hint: if this import works and credentials are set, we're likely good
    app = modal.App.lookup("formrecap-lora-verify", create_if_missing=True)
    # A full GPU call would cost $; just validate auth for now
    os.environ["MODAL_TOKEN_ID"]
    os.environ["MODAL_TOKEN_SECRET"]


# 3. CF Workers AI base model availability
def check_cf_base():
    account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    token = os.environ["CLOUDFLARE_API_TOKEN"]
    r = httpx.post(
        f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/meta/llama-3.2-3b-instruct",
        headers={"Authorization": f"Bearer {token}"},
        json={"messages": [{"role": "user", "content": "hello"}]},
        timeout=30,
    )
    assert r.status_code == 200, f"Base model call failed: {r.status_code} {r.text}"


# 4. CF Workers AI LoRA endpoint name + logprobs
def check_cf_lora_endpoint():
    account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    token = os.environ["CLOUDFLARE_API_TOKEN"]
    # Try the -lora suffix pattern first (historical CF convention)
    for model in [
        "@cf/meta/llama-3.2-3b-instruct-lora",
        "@cf/meta/llama-3.2-3b-instruct",
    ]:
        r = httpx.post(
            f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "logprobs": True,
                "top_logprobs": 5,
            },
            timeout=30,
        )
        body = {}
        try:
            body = r.json()
        except Exception:
            pass
        result = body.get("result")
        has_logprobs = bool(result.get("logprobs")) if isinstance(result, dict) else False
        rprint(
            f"  [dim]model={model} status={r.status_code} logprobs_in_response={has_logprobs}[/dim]"
        )
    # Expected: at least one status 200. logprobs likely False for the LoRA endpoint.


# 5. Turnstile sitekey generation (just auth check)
def check_cf_auth_token():
    account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    token = os.environ["CLOUDFLARE_API_TOKEN"]
    r = httpx.get(
        f"https://api.cloudflare.com/client/v4/accounts/{account}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    assert r.status_code == 200, f"CF auth failed: {r.status_code}"


# 6. Anthropic API
def check_anthropic():
    from anthropic import Anthropic

    client = Anthropic()
    r = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": "ping"}],
    )
    assert r.content, "Empty response"


checks = [
    ("HuggingFace Llama 3.2 3B license", check_hf),
    ("Modal auth", check_modal),
    ("CF Workers AI base model", check_cf_base),
    ("CF Workers AI LoRA endpoint probe", check_cf_lora_endpoint),
    ("CF account auth token", check_cf_auth_token),
    ("Anthropic API", check_anthropic),
]

for name, fn in checks:
    check(name, fn)

table = Table(title="Day-1 Verification")
table.add_column("Check")
table.add_column("Status")
table.add_column("Details")
for name, (status, detail) in results.items():
    colour = "green" if status == "PASS" else "red"
    table.add_row(name, f"[{colour}]{status}[/{colour}]", detail[:80])
rprint(table)

if any(s == "FAIL" for s, _ in results.values()):
    sys.exit(1)
