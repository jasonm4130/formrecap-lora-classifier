import { issueToken } from "../hmac";

export interface Env {
  HMAC_SECRET: string;
  TURNSTILE_SECRET_KEY: string;
}

async function verifyTurnstile(secret: string, token: string): Promise<boolean> {
  const body = new FormData();
  body.append("secret", secret);
  body.append("response", token);
  const r = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", { method: "POST", body });
  const json: any = await r.json();
  return Boolean(json.success);
}

export async function handleToken(env: Env, request: Request): Promise<Response> {
  if (request.method !== "GET") return new Response("Method not allowed", { status: 405 });

  const tsToken = request.headers.get("X-Turnstile-Token");
  if (!tsToken || !(await verifyTurnstile(env.TURNSTILE_SECRET_KEY, tsToken))) {
    return new Response("Turnstile required", { status: 403 });
  }

  const token = await issueToken(env.HMAC_SECRET, 600);
  return new Response(JSON.stringify({ token }), {
    headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
  });
}
