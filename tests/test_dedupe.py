from formrecap_lora.data.dedupe import dedupe_exact, is_near_duplicate


def test_dedupe_exact_removes_matching_events():
    records = [
        {"events": "focus:email, input:email(x3), exit", "code": 1, "reason": "a", "confidence": 0.5},
        {"events": "focus:email, input:email(x3), exit", "code": 1, "reason": "b", "confidence": 0.7},
        {"events": "focus:name, exit", "code": 2, "reason": "c", "confidence": 0.6},
    ]
    result = dedupe_exact(records)
    assert len(result) == 2
    assert result[0]["events"] == "focus:email, input:email(x3), exit"
    assert result[1]["events"] == "focus:name, exit"


def test_dedupe_exact_empty_list():
    assert dedupe_exact([]) == []


def test_dedupe_exact_all_unique_passes_through():
    records = [
        {"events": "a", "code": 1, "reason": "r", "confidence": 0.5},
        {"events": "b", "code": 2, "reason": "r", "confidence": 0.5},
    ]
    assert len(dedupe_exact(records)) == 2


def test_is_near_duplicate_identical_embeddings():
    e1 = [1.0, 0.0, 0.0]
    e2 = [1.0, 0.0, 0.0]
    assert is_near_duplicate(e1, e2, threshold=0.95) is True


def test_is_near_duplicate_orthogonal():
    e1 = [1.0, 0.0, 0.0]
    e2 = [0.0, 1.0, 0.0]
    assert is_near_duplicate(e1, e2, threshold=0.95) is False


def test_is_near_duplicate_just_below_threshold():
    e1 = [1.0, 0.0]
    e2 = [0.94, 0.34]  # cos sim ~0.94
    assert is_near_duplicate(e1, e2, threshold=0.95) is False
