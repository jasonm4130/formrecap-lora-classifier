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

    let response: Response;
    if (url.pathname === "/classify") {
      response = await handleClassify(env, request);
    } else if (url.pathname === "/issue-token") {
      response = await handleToken(env, request);
    } else {
      response = new Response("Not found", { status: 404 });
    }

    // Ensure CORS headers on ALL responses (not just preflight)
    const corsResponse = new Response(response.body, response);
    corsResponse.headers.set("Access-Control-Allow-Origin", "*");
    return corsResponse;
  },
};
