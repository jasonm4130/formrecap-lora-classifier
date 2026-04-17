"""Hand-crafted few-shot primers for each class. 2-3 per class."""

from typing import TypedDict, Literal

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
    # 1. validation_error
    {
        "events": "focus:email, input:email(x8), blur:email(invalid_format), focus:email, input:email(x4), blur:email(invalid_format), exit",
        "code": 1,
        "reason": "User repeatedly attempted to correct email format, hit the same validation twice, then exited without resolving.",
        "confidence": 0.88,
    },
    {
        "events": "focus:phone, input:phone(x10), blur:phone(required), focus:phone, input:phone(x2), blur:phone(required), scroll:page(800ms), exit",
        "code": 1,
        "reason": "Phone field flagged as required, user retried twice, scrolled briefly, gave up.",
        "confidence": 0.82,
    },
    # 2. distraction
    {
        "events": "focus:email, input:email(x5), scroll:page(4200ms), scroll:page(8800ms), exit",
        "code": 2,
        "reason": "User started typing email then paused for extended periods suggesting task-switching before exiting mid-form.",
        "confidence": 0.68,
    },
    {
        "events": "focus:name, input:name(x3), scroll:page(6000ms), exit",
        "code": 2,
        "reason": "Brief engagement with name field followed by long idle scroll and exit, typical of interrupted focus.",
        "confidence": 0.62,
    },
    # 3. comparison_shopping
    {
        "events": "scroll:page(1200ms), focus:price, blur:price, focus:features, blur:features, scroll:page(3400ms), focus:price, scroll:page(2100ms), exit",
        "code": 3,
        "reason": "User repeatedly focused on price and features without filling any input fields, scrolled extensively, exited without engaging.",
        "confidence": 0.78,
    },
    {
        "events": "scroll:page(2000ms), focus:plan_pro, blur:plan_pro, focus:plan_basic, blur:plan_basic, scroll:page(1500ms), exit",
        "code": 3,
        "reason": "Browsing across plan options without committing to any, signalling evaluation.",
        "confidence": 0.74,
    },
    # 4. accidental_exit
    {
        "events": "focus:email, input:email(x18), focus:name, input:name(x6), exit",
        "code": 4,
        "reason": "Strong progress across multiple fields then abrupt exit without validation errors or idle time — plausibly accidental navigation.",
        "confidence": 0.55,
    },
    {
        "events": "focus:name, input:name(x8), focus:email, input:email(x14), focus:password, input:password(x9), exit",
        "code": 4,
        "reason": "User filled three consecutive fields rapidly, no errors, ended mid-input — signature of back-button or tab close.",
        "confidence": 0.60,
    },
    # 5. bot
    {
        "events": "input:email(x32), input:password(x48), submit",
        "code": 5,
        "reason": "No focus events, instantaneous character entry, no scrolling — consistent with automated form filling.",
        "confidence": 0.92,
    },
    {
        "events": "input:email(x28), input:name(x20), input:password(x35), submit",
        "code": 5,
        "reason": "Direct input bursts without any focus, blur, or scroll signals — typical automation footprint.",
        "confidence": 0.89,
    },
    # 6. committed_leave
    {
        "events": "scroll:page(3000ms), focus:email, blur:email, scroll:page(2000ms), exit",
        "code": 6,
        "reason": "Brief inspection of email field and exit without typing — decision to not proceed rather than abandonment mid-engagement.",
        "confidence": 0.65,
    },
    {
        "events": "scroll:page(1500ms), exit",
        "code": 6,
        "reason": "Glanced at form, did not engage any field, left deliberately.",
        "confidence": 0.72,
    },
]


def get_primers_for_class(code: ClassCode) -> list[Primer]:
    return [p for p in PRIMERS if p["code"] == code]
