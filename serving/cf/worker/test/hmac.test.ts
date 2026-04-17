import { describe, expect, it } from "vitest";
import { issueToken, verifyToken } from "../src/hmac";

const SECRET = "test-secret-key-long-enough";

describe("HMAC tokens", () => {
  it("verifies a freshly issued token", async () => {
    const token = await issueToken(SECRET, 600);
    const result = await verifyToken(SECRET, token);
    expect(result.ok).toBe(true);
  });

  it("rejects a tampered token", async () => {
    const token = await issueToken(SECRET, 600);
    const tampered = token.slice(0, -2) + "xx";
    const result = await verifyToken(SECRET, tampered);
    expect(result.ok).toBe(false);
  });

  it("rejects an expired token", async () => {
    const token = await issueToken(SECRET, -10);
    const result = await verifyToken(SECRET, token);
    expect(result.ok).toBe(false);
  });

  it("rejects a token signed with a different secret", async () => {
    const token = await issueToken(SECRET, 600);
    const result = await verifyToken("different-secret", token);
    expect(result.ok).toBe(false);
  });
});
