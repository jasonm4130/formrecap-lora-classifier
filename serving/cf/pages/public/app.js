const API_BASE = "https://formrecap-classify.jasonm4130.workers.dev";
let demoToken = null;

async function onTurnstileVerified(tsToken) {
  const r = await fetch(`${API_BASE}/issue-token`, { headers: { "X-Turnstile-Token": tsToken } });
  const { token } = await r.json();
  demoToken = token;
  document.getElementById("classify").disabled = false;
}
window.onTurnstileVerified = onTurnstileVerified;

document.getElementById("try-example").addEventListener("click", () => {
  document.getElementById("events").value =
    "focus:email, input:email(x8), blur:email(invalid_format), focus:email, input:email(x4), blur:email(invalid_format), exit";
});

document.getElementById("classify").addEventListener("click", async () => {
  const events = document.getElementById("events").value.trim();
  const result = document.getElementById("result");
  result.textContent = "Classifying...";
  try {
    const r = await fetch(`${API_BASE}/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Demo-Token": demoToken },
      body: JSON.stringify({ events }),
    });
    const json = await r.json();
    result.textContent = JSON.stringify(json, null, 2);
  } catch (e) {
    result.textContent = "Error: " + e.message;
  }
});
