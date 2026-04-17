import { handleClassify, Env as ClassifyEnv } from "./handlers/classify";
import { handleToken } from "./handlers/token";

interface Env extends ClassifyEnv {
  AI: any;
  STATE: KVNamespace;
  HMAC_SECRET: string;
  TURNSTILE_SECRET_KEY: string;
  LORA_FINETUNE_ID: string;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, X-Demo-Token, X-Turnstile-Token",
        },
      });
    }

    if (url.pathname === "/classify") return handleClassify(env, request);
    if (url.pathname === "/issue-token") return handleToken(env, request);
    return new Response("Not found", { status: 404 });
  },
};
