const API_BASE = "https://formrecap-classify.jasonm4130.workers.dev";
let demoToken = null;

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

// Example buttons
document.querySelectorAll(".example-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".example-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("events").value = btn.dataset.trace;
  });
});

// Classify
document.getElementById("classify").addEventListener("click", async () => {
  const events = document.getElementById("events").value.trim();
  if (!events) return;

  const btn = document.getElementById("classify");
  const btnText = btn.querySelector(".btn-text");
  const btnLoading = btn.querySelector(".btn-loading");

  btn.disabled = true;
  btnText.hidden = true;
  btnLoading.hidden = false;

  const resultSection = document.getElementById("result-section");
  resultSection.hidden = true;

  try {
    const r = await fetch(`${API_BASE}/classify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Demo-Token": demoToken,
      },
      body: JSON.stringify({ events }),
    });
    const json = await r.json();

    // Display structured result
    const className = CLASS_NAMES[json.code] || "unknown";
    document.getElementById("result-class").textContent =
      `${json.code} — ${className}`;
    document.getElementById("result-confidence").textContent =
      json.confidence ? `confidence: ${(json.confidence * 100).toFixed(0)}%` : "";
    document.getElementById("result-reason").textContent =
      json.reason || json.raw || "No reason provided";
    document.getElementById("result-json").textContent =
      JSON.stringify(json, null, 2);

    resultSection.hidden = false;
  } catch (e) {
    document.getElementById("result-class").textContent = "Error";
    document.getElementById("result-confidence").textContent = "";
    document.getElementById("result-reason").textContent = e.message;
    document.getElementById("result-json").textContent = "";
    resultSection.hidden = false;
  } finally {
    btn.disabled = false;
    btnText.hidden = false;
    btnLoading.hidden = true;
  }
});
