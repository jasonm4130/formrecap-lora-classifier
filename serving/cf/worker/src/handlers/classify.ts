import { verifyToken } from "../hmac";
import { validateEventTrace } from "../validate";

const DAILY_NEURON_BUDGET = 90_000;
const ESTIMATED_NEURONS_PER_CALL = 1;

const SYSTEM_PROMPT =
  "You analyse form interaction event sequences and classify the likely abandonment reason. " +
  "Respond with a digit class code 1-6 on the first line, then a JSON object on the second line " +
  "with class, reason, and confidence fields. " +
  "Classes: 1=validation_error, 2=distraction, 3=comparison_shopping, " +
  "4=accidental_exit, 5=bot, 6=committed_leave.";

function parseModelOutput(text: string): Record<string, unknown> {
  const [first, ...restArr] = text.trim().split("\n");
  const rest = restArr.join("\n").trim();
  const codeMatch = first.match(/([1-6])/);
  const code = codeMatch ? parseInt(codeMatch[1], 10) : null;
  let obj: Record<string, unknown> = {};
  try { obj = JSON.parse(rest); } catch { /* fallback */ }
  return { code, ...obj, raw: text };
}

export interface Env {
  AI: any;
  STATE: KVNamespace;
  HMAC_SECRET: string;
  TURNSTILE_SECRET_KEY: string;
  LORA_FINETUNE_ID: string;
}

export async function handleClassify(env: Env, request: Request): Promise<Response> {
  if (request.method !== "POST") return new Response("POST only", { status: 405 });

  // Kill switch
  const enabled = await env.STATE.get("demo_enabled");
  if (enabled === "false") return new Response("Demo disabled", { status: 503 });

  // Demo token
  const token = request.headers.get("X-Demo-Token");
  if (!token || !(await verifyToken(env.HMAC_SECRET, token)).ok) {
    return new Response("Invalid demo token", { status: 403 });
  }

  // Budget
  const today = new Date().toISOString().slice(0, 10).replace(/-/g, "");
  const key = `neurons_${today}`;
  const spentStr = (await env.STATE.get(key)) ?? "0";
  const spent = parseInt(spentStr, 10);
  if (spent >= DAILY_NEURON_BUDGET) {
    return new Response("Daily budget exhausted", { status: 503 });
  }

  // Parse + validate
  const body = (await request.json().catch(() => ({}))) as { events?: string };
  if (!body.events) return new Response("missing events", { status: 400 });
  const check = validateEventTrace(body.events);
  if (!check.ok) return new Response(JSON.stringify({ error: check.reason }), { status: 400 });

  // Call Workers AI — Mistral uses [INST] tags, system merged into user turn
  const messages = [
    { role: "user", content: `${SYSTEM_PROMPT}\n\nEvents: ${body.events}` },
  ];
  const aiResp: any = await env.AI.run("@cf/mistral/mistral-7b-instruct-v0.2-lora", { messages, lora: env.LORA_FINETUNE_ID });
  const text = aiResp.response as string;

  const parsed = parseModelOutput(text);

  // Increment counter
  await env.STATE.put(key, String(spent + ESTIMATED_NEURONS_PER_CALL), { expirationTtl: 60 * 60 * 48 });

  return new Response(JSON.stringify(parsed), {
    headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
  });
}
