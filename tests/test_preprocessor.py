import pytest
from formrecap_lora.data.preprocessor import normalize_events, EventTrace


def test_single_focus_event():
    events = [{"type": "focus", "field": "email", "ts": 1200}]
    assert normalize_events(events) == "focus:email"


def test_repeated_input_compressed():
    events = [
        {"type": "focus", "field": "email", "ts": 1200},
        *[{"type": "input", "field": "email", "ts": 1200 + i * 50} for i in range(8)],
        {"type": "blur", "field": "email", "ts": 1800, "validation": "invalid_format"},
    ]
    assert normalize_events(events) == "focus:email, input:email(x8), blur:email(invalid_format)"


def test_exit_terminates():
    events = [
        {"type": "scroll", "field": "page", "ts": 1000, "duration_ms": 1200},
        {"type": "exit", "ts": 2200},
    ]
    assert normalize_events(events) == "scroll:page(1200ms), exit"


def test_validation_in_blur():
    events = [{"type": "blur", "field": "email", "ts": 100, "validation": "required"}]
    assert normalize_events(events) == "blur:email(required)"


def test_empty_list_returns_empty_string():
    assert normalize_events([]) == ""


def test_unknown_event_type_raises():
    with pytest.raises(ValueError, match="Unknown event type"):
        normalize_events([{"type": "bogus", "field": "x", "ts": 0}])


def test_missing_ts_raises():
    with pytest.raises(KeyError):
        normalize_events([{"type": "focus", "field": "email"}])


def test_multiple_inputs_with_different_field_not_compressed():
    events = [
        {"type": "input", "field": "email", "ts": 100},
        {"type": "input", "field": "name", "ts": 200},
    ]
    assert normalize_events(events) == "input:email, input:name"


def test_scroll_rounds_duration_to_ms():
    events = [{"type": "scroll", "field": "page", "ts": 0, "duration_ms": 3456}]
    assert normalize_events(events) == "scroll:page(3456ms)"
