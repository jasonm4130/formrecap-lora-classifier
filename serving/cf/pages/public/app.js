const API_BASE = "https://formrecap-classify.jasonm4130.workers.dev";
let demoToken = null;
let selectedModel = "cf-lora";

const CLASS_NAMES = {
  1: "validation_error",
  2: "distraction",
  3: "comparison_shopping",
  4: "accidental_exit",
  5: "bot",
  6: "committed_leave",
};

async function onTurnstileVerified(tsToken) {
  try {
    const r = await fetch(`${API_BASE}/issue-token`, {
      headers: { "X-Turnstile-Token": tsToken },
    });
    if (!r.ok) throw new Error(`Token issue failed: ${r.status}`);
    const { token } = await r.json();
    demoToken = token;
    document.getElementById("classify").disabled = false;
  } catch (e) {
    console.error("Token issue failed:", e);
  }
}
window.onTurnstileVerified = onTurnstileVerified;

// Model selector
document.querySelectorAll(".model-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".model-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    selectedModel = btn.dataset.model;
  });
});

// Example buttons
document.querySelectorAll(".ex-btn:not(.model-btn)").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".ex-btn:not(.model-btn)").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("events").value = btn.dataset.trace;
  });
});

// Classify
document.getElementById("classify").addEventListener("click", async () => {
  const events = document.getElementById("events").value.trim();
  if (!events) return;

  const btn = document.getElementById("classify");
  const btnText = document.getElementById("btn-text");
  const btnLoading = document.getElementById("btn-loading");

  btn.disabled = true;
  btnText.hidden = true;
  btnLoading.hidden = false;
  document.getElementById("result").hidden = true;

  try {
    const r = await fetch(`${API_BASE}/classify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Demo-Token": demoToken,
      },
      body: JSON.stringify({ events, model: selectedModel }),
    });
    const json = await r.json();

    const className = CLASS_NAMES[json.code] || "unknown";
    document.getElementById("r-class").textContent = `${json.code} — ${className}`;

    const parts = [];
    if (json.confidence) parts.push(`confidence: ${(json.confidence * 100).toFixed(0)}%`);
    if (json.latency_ms) parts.push(`${json.latency_ms}ms`);
    if (json.model) parts.push(json.model);
    document.getElementById("r-conf").textContent = parts.join(" · ");

    document.getElementById("r-reason").textContent =
      json.reason || json.raw || "No reason provided";
    document.getElementById("r-json").textContent = JSON.stringify(json, null, 2);
    document.getElementById("result").hidden = false;
  } catch (e) {
    document.getElementById("r-class").textContent = "Error";
    document.getElementById("r-conf").textContent = "";
    document.getElementById("r-reason").textContent = e.message;
    document.getElementById("r-json").textContent = "";
    document.getElementById("result").hidden = false;
  } finally {
    btn.disabled = false;
    btnText.hidden = false;
    btnLoading.hidden = true;
  }
});
