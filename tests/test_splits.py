from formrecap_lora.data.splits import split_stratified


def test_split_stratified_respects_class_balance():
    records = []
    for code in [1, 2, 3, 4, 5, 6]:
        for i in range(20):
            records.append({"events": f"e{code}_{i}", "code": code, "reason": "", "confidence": 0.5})
    train, val, test = split_stratified(records, val_frac=0.1, test_frac=0.1, seed=42)
    assert len(train) == 120 * 0.8
    assert len(val) == 120 * 0.1
    assert len(test) == 120 * 0.1
    # Each split should have at least one of each class
    for split in [train, val, test]:
        codes = {r["code"] for r in split}
        assert codes == {1, 2, 3, 4, 5, 6}


def test_split_is_deterministic_with_seed():
    records = [{"events": f"e{i}", "code": (i % 6) + 1, "reason": "", "confidence": 0.5} for i in range(120)]
    s1 = split_stratified(records, val_frac=0.1, test_frac=0.1, seed=42)
    s2 = split_stratified(records, val_frac=0.1, test_frac=0.1, seed=42)
    assert [r["events"] for r in s1[0]] == [r["events"] for r in s2[0]]
