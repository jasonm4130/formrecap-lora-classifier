export interface TokenPayload { exp: number; }

async function hmacSha256(secret: string, data: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(data));
  return btoa(String.fromCharCode(...new Uint8Array(sig))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export async function issueToken(secret: string, ttlSeconds: number): Promise<string> {
  const exp = Math.floor(Date.now() / 1000) + ttlSeconds;
  const payload = btoa(JSON.stringify({ exp })).replace(/=+$/, "");
  const sig = await hmacSha256(secret, payload);
  return `${payload}.${sig}`;
}

export async function verifyToken(secret: string, token: string): Promise<{ ok: boolean; reason?: string }> {
  const parts = token.split(".");
  if (parts.length !== 2) return { ok: false, reason: "format" };
  const [payload, sig] = parts;
  const expected = await hmacSha256(secret, payload);
  if (sig !== expected) return { ok: false, reason: "signature" };
  try {
    const obj: TokenPayload = JSON.parse(atob(payload));
    if (obj.exp < Math.floor(Date.now() / 1000)) return { ok: false, reason: "expired" };
  } catch {
    return { ok: false, reason: "payload" };
  }
  return { ok: true };
}
