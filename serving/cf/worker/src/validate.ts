const MAX_BYTES = 2048;
const TOKEN_RE = /^((focus|blur|input|scroll|exit|submit)(:[a-z_]+(\([^()]*\))?)?)(,\s*((focus|blur|input|scroll|exit|submit)(:[a-z_]+(\([^()]*\))?)?))*$/;

export function validateEventTrace(input: string): { ok: boolean; reason?: string } {
  if (!input) return { ok: false, reason: "empty" };
  if (new TextEncoder().encode(input).byteLength > MAX_BYTES) return { ok: false, reason: "too_large" };
  if (!TOKEN_RE.test(input.trim())) return { ok: false, reason: "malformed" };
  return { ok: true };
}
