"""Hand-crafted few-shot primers for each class. 4-5 per class, maximising diversity."""

from typing import Literal, TypedDict

ClassCode = Literal[1, 2, 3, 4, 5, 6]
CLASS_NAMES: dict[ClassCode, str] = {
    1: "validation_error",
    2: "distraction",
    3: "comparison_shopping",
    4: "accidental_exit",
    5: "bot",
    6: "committed_leave",
}


class Primer(TypedDict):
    events: str  # terse token format
    code: ClassCode
    reason: str
    confidence: float


PRIMERS: list[Primer] = [
    # ── 1. validation_error ──────────────────────────────────────────────────
    # Signature: repeated focus/blur cycles on the SAME field with validation failures,
    # edit-check-edit loops, no forward progress past the error point.
    {
        "events": "focus:email, input:email(x8), blur:email(invalid_format), focus:email, input:email(x4), blur:email(invalid_format), exit",
        "code": 1,
        "reason": "User repeatedly attempted to correct email format, hit the same validation twice, then exited without resolving.",
        "confidence": 0.88,
    },
    {
        "events": "focus:name, input:name(x6), blur:name, focus:dob, input:dob(x8), blur:dob(invalid_date), focus:dob, input:dob(x6), blur:dob(invalid_date), focus:dob, input:dob(x10), blur:dob(invalid_date), exit",
        "code": 1,
        "reason": "Date of birth field rejected three times — likely a format mismatch (DD/MM/YYYY vs MM/DD/YYYY). User progressed past name but stalled entirely at DOB.",
        "confidence": 0.85,
    },
    {
        "events": "focus:card_number, input:card_number(x16), blur:card_number(invalid_luhn), focus:card_number, input:card_number(x16), blur:card_number(invalid_luhn), scroll:page(1200ms), exit",
        "code": 1,
        "reason": "Card number failed Luhn check twice. User likely mistyped a digit and couldn't identify which one. Brief scroll before abandoning checkout.",
        "confidence": 0.90,
    },
    {
        "events": "focus:postcode, input:postcode(x4), blur:postcode(invalid_format), focus:postcode, input:postcode(x5), blur:postcode(invalid_format), focus:postcode, input:postcode(x6), blur:postcode(invalid_format), exit",
        "code": 1,
        "reason": "Postcode rejected three times for format — possibly entering a non-US zip in a US-only field, or including spaces in a no-spaces validator.",
        "confidence": 0.82,
    },
    {
        "events": "focus:username, input:username(x5), blur:username(taken), focus:username, input:username(x8), blur:username(taken), focus:username, input:username(x12), blur:username(taken), exit",
        "code": 1,
        "reason": "Three unique username attempts all returned 'taken'. User gave up rather than finding an available name — common on high-traffic signup forms.",
        "confidence": 0.86,
    },
    # ── 2. distraction ───────────────────────────────────────────────────────
    # Signature: normal field progression then a LONG idle gap (60-300s+),
    # partial completion (30-70% filled), no errors.
    {
        "events": "focus:name, input:name(x7), blur:name, focus:email, input:email(x12), blur:email, scroll:page(45000ms), exit",
        "code": 2,
        "reason": "Completed name and email normally, then 45 seconds of idle before exit — consistent with tab-switching to another task and not returning.",
        "confidence": 0.72,
    },
    {
        "events": "focus:first_name, input:first_name(x5), blur:first_name, focus:last_name, input:last_name(x7), blur:last_name, focus:email, input:email(x3), scroll:page(120000ms), exit",
        "code": 2,
        "reason": "Started email entry then 2 minutes idle. Partial email input (only 3 keystrokes) followed by extended absence suggests a phone call or notification interrupted.",
        "confidence": 0.78,
    },
    {
        "events": "focus:company, input:company(x9), blur:company, focus:role, input:role(x5), blur:role, focus:phone, input:phone(x4), scroll:page(85000ms), scroll:page(35000ms), exit",
        "code": 2,
        "reason": "Progressive form fill through three fields, then two long idle periods (85s + 35s) — user was pulled away, briefly returned, then left again.",
        "confidence": 0.70,
    },
    {
        "events": "focus:address, input:address(x18), blur:address, focus:city, input:city(x6), blur:city, focus:state, scroll:page(180000ms), exit",
        "code": 2,
        "reason": "Deep form progress (address, city) then 3-minute gap at state field. The extended idle with no further interaction is characteristic of a context switch.",
        "confidence": 0.75,
    },
    # ── 3. comparison_shopping ───────────────────────────────────────────────
    # Signature: high scroll-to-input ratio, focus/blur without typing,
    # reading pricing/features/terms, minimal or zero field input.
    {
        "events": "scroll:page(2800ms), focus:pricing_table, blur:pricing_table, scroll:page(1500ms), focus:feature_list, blur:feature_list, scroll:page(3200ms), focus:pricing_table, blur:pricing_table, exit",
        "code": 3,
        "reason": "Bounced between pricing and features sections without typing anything. Returned to pricing — evaluating cost vs value before deciding against.",
        "confidence": 0.80,
    },
    {
        "events": "focus:plan_starter, blur:plan_starter, focus:plan_professional, blur:plan_professional, focus:plan_enterprise, blur:plan_enterprise, scroll:page(4500ms), focus:plan_professional, blur:plan_professional, exit",
        "code": 3,
        "reason": "Cycled through three plan tiers, revisited professional tier after scrolling — comparing options systematically without committing.",
        "confidence": 0.76,
    },
    {
        "events": "scroll:page(1800ms), focus:shipping_cost, blur:shipping_cost, focus:delivery_estimate, blur:delivery_estimate, scroll:page(2400ms), focus:return_policy, blur:return_policy, exit",
        "code": 3,
        "reason": "Reviewed shipping, delivery, and return policy without filling any fields — evaluating purchase terms before deciding not to proceed.",
        "confidence": 0.82,
    },
    {
        "events": "focus:email, input:email(x14), blur:email, scroll:page(5000ms), focus:total_price, blur:total_price, scroll:page(3000ms), focus:coupon_code, blur:coupon_code, exit",
        "code": 3,
        "reason": "Entered email (showing some intent) but then spent time on price and coupon fields without engaging — checking if there's a discount before committing.",
        "confidence": 0.65,
    },
    # ── 4. accidental_exit ───────────────────────────────────────────────────
    # Signature: steady progress, good input velocity, multiple fields filled,
    # NO errors, NO idle time, abrupt termination mid-flow.
    {
        "events": "focus:email, input:email(x18), blur:email, focus:name, input:name(x6), exit",
        "code": 4,
        "reason": "Strong progress across two fields with no errors or pauses, then abrupt exit — likely back button or accidental tab close.",
        "confidence": 0.55,
    },
    {
        "events": "focus:first_name, input:first_name(x5), blur:first_name, focus:last_name, input:last_name(x8), blur:last_name, focus:email, input:email(x14), blur:email, focus:phone, input:phone(x6), exit",
        "code": 4,
        "reason": "Filled four fields rapidly with zero errors, mid-phone-entry exit. No deceleration or hesitation — consistent with accidental navigation.",
        "confidence": 0.62,
    },
    {
        "events": "focus:address, input:address(x22), blur:address, focus:city, input:city(x8), blur:city, focus:state, input:state(x2), exit",
        "code": 4,
        "reason": "Deep in the address section, only 2 keystrokes into state field — interrupted mid-input with no reason to voluntarily stop. Accidental close.",
        "confidence": 0.58,
    },
    {
        "events": "focus:name, input:name(x7), blur:name, focus:email, input:email(x16), blur:email, focus:password, input:password(x12), blur:password, focus:confirm_password, input:confirm_password(x4), exit",
        "code": 4,
        "reason": "Nearly completed signup (name, email, password done, confirm started) then exited 4 keystrokes in. No validation errors, no pauses — accidental.",
        "confidence": 0.60,
    },
    # ── 5. bot ───────────────────────────────────────────────────────────────
    # Signature: NO focus events (or instant focus-input pairs), no scroll,
    # no blur events, input bursts, strict field order, sub-3s total time.
    {
        "events": "input:email(x32), input:password(x48), submit",
        "code": 5,
        "reason": "No focus events, no blur, no scrolling — direct input injection followed by submit. Classic automated form fill.",
        "confidence": 0.94,
    },
    {
        "events": "input:name(x15), input:email(x28), input:phone(x10), input:message(x200), submit",
        "code": 5,
        "reason": "Four fields filled via direct input with no human interaction signals (no focus, blur, scroll). Message field has 200 keystrokes — likely spam payload.",
        "confidence": 0.92,
    },
    {
        "events": "input:first_name(x8), input:last_name(x10), input:email(x24), input:company(x12), input:phone(x10), submit",
        "code": 5,
        "reason": "Five fields in strict DOM order, no focus/blur events, no pauses. Programmatic form submission — no human browses a form this linearly.",
        "confidence": 0.90,
    },
    {
        "events": "input:username(x12), input:password(x20), input:email(x26), submit",
        "code": 5,
        "reason": "Rapid input injection across three fields with zero interaction overhead. No focus, blur, or scroll events — automated credential stuffing pattern.",
        "confidence": 0.93,
    },
    # ── 6. committed_leave ───────────────────────────────────────────────────
    # Signature: browsed form, maybe focused fields without typing, possibly
    # scrolled to terms/pricing. Deliberate decision not to proceed.
    {
        "events": "scroll:page(3000ms), focus:email, blur:email, scroll:page(2000ms), exit",
        "code": 6,
        "reason": "Scrolled the form, clicked the email field but typed nothing, scrolled more, left. Evaluating form length/requirements before deciding against it.",
        "confidence": 0.65,
    },
    {
        "events": "scroll:page(1500ms), exit",
        "code": 6,
        "reason": "Brief glance at the form — 1.5 seconds of scrolling, never engaged any field. Immediate deliberate departure.",
        "confidence": 0.72,
    },
    {
        "events": "focus:name, input:name(x5), blur:name, focus:email, blur:email, scroll:page(4000ms), focus:terms, blur:terms, exit",
        "code": 6,
        "reason": "Started name field, inspected email without typing, then scrolled down to read terms and left. The terms review suggests a conscious decision against proceeding.",
        "confidence": 0.68,
    },
    {
        "events": "scroll:page(2500ms), focus:required_documents, blur:required_documents, scroll:page(1800ms), focus:upload_id, blur:upload_id, exit",
        "code": 6,
        "reason": "Reviewed required documents section and upload requirements without engaging — realized the form requires materials they don't have. Deliberate exit.",
        "confidence": 0.75,
    },
    {
        "events": "focus:email, input:email(x12), blur:email, focus:phone, blur:phone, focus:company_size, blur:company_size, scroll:page(3500ms), exit",
        "code": 6,
        "reason": "Entered email but then inspected remaining fields (phone, company size) without typing. Scrolled to see full form length and left — decided it's not worth the effort.",
        "confidence": 0.62,
    },
]


def get_primers_for_class(code: ClassCode) -> list[Primer]:
    return [p for p in PRIMERS if p["code"] == code]
