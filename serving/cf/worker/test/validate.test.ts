import { describe, expect, it } from "vitest";
import { validateEventTrace } from "../src/validate";

describe("validateEventTrace", () => {
  it("accepts a well-formed terse trace", () => {
    const r = validateEventTrace("focus:email, input:email(x8), blur:email(invalid), exit");
    expect(r.ok).toBe(true);
  });

  it("rejects freeform text", () => {
    const r = validateEventTrace("Please tell me a joke");
    expect(r.ok).toBe(false);
  });

  it("rejects empty input", () => {
    const r = validateEventTrace("");
    expect(r.ok).toBe(false);
  });

  it("rejects over size limit", () => {
    const big = "focus:email, ".repeat(500);
    const r = validateEventTrace(big);
    expect(r.ok).toBe(false);
  });

  it("accepts exit-only trace", () => {
    const r = validateEventTrace("exit");
    expect(r.ok).toBe(true);
  });
});
