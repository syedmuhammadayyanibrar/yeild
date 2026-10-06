from loop.rng import get_rng
from loop.schema import LANGUAGES, REPLY_CLASSES
from loop.simulate.replies import (
    SPLIT_CORE_IDS,
    TEMPLATES_BY_LANG,
    build_hard_dataset,
    build_test_dataset,
)


def test_no_core_shared_across_splits() -> None:
    train_cores = set(SPLIT_CORE_IDS["train"])
    calib_cores = set(SPLIT_CORE_IDS["calibration"])
    test_cores = set(SPLIT_CORE_IDS["test"])

    assert len(train_cores) == 6
    assert len(calib_cores) == 3
    assert len(test_cores) == 3

    assert train_cores.isdisjoint(calib_cores)
    assert train_cores.isdisjoint(test_cores)
    assert calib_cores.isdisjoint(test_cores)

    for lang in LANGUAGES:
        for cls in REPLY_CLASSES:
            cores = TEMPLATES_BY_LANG[lang][cls]
            assert len(cores) >= 12

def test_test_set_size_and_stratification() -> None:
    rng = get_rng(42)
    test_dataset = build_test_dataset(rng)
    assert len(test_dataset) == 600

    counts: dict[tuple[str, str], int] = {}
    for item in test_dataset:
        key = (item["true_class"], item["true_language"])
        counts[key] = counts.get(key, 0) + 1
        assert item["core_id"] in SPLIT_CORE_IDS["test"]

    for cls in REPLY_CLASSES:
        for lang in LANGUAGES:
            assert counts.get((cls, lang), 0) == 15

def test_hard_set_composition() -> None:
    rng = get_rng(42)
    hard_dataset = build_hard_dataset(rng)
    assert len(hard_dataset) == 200

    injection_items = [item for item in hard_dataset if item.get("is_injection")]
    assert len(injection_items) == 20

    test_cores = set(SPLIT_CORE_IDS["test"])
    for item in hard_dataset:
        assert item["core_id"] in test_cores
