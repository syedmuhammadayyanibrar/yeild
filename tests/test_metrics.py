import numpy as np

from loop.evaluation.metrics import (
    compute_dr,
    compute_f1_score,
    compute_ips,
    compute_snips,
)
from loop.reply_intel.calibration import compute_ece


def test_hand_computed_f1() -> None:
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "a", "b"]
    classes = ["a", "b"]

    macro_f1, per_class = compute_f1_score(y_true, y_pred, classes=classes)
    assert abs(macro_f1 - 0.5) < 1e-6
    assert abs(per_class["a"]["precision"] - 0.5) < 1e-6
    assert abs(per_class["a"]["recall"] - 0.5) < 1e-6
    assert abs(per_class["a"]["f1"] - 0.5) < 1e-6
    assert abs(per_class["b"]["precision"] - 0.5) < 1e-6
    assert abs(per_class["b"]["recall"] - 0.5) < 1e-6
    assert abs(per_class["b"]["f1"] - 0.5) < 1e-6

def test_hand_computed_ece() -> None:
    probs = np.array(
        [
            [0.8, 0.2],
            [0.6, 0.4],
        ],
        dtype=float,
    )
    true_indices = np.array([0, 1], dtype=int)
    ece, _ = compute_ece(probs, true_indices, n_bins=2)
    assert abs(ece - 0.2) < 1e-6

def test_hand_computed_ips_and_snips() -> None:
    rewards = np.array([1.0, 0.0], dtype=float)
    behavior = np.array([0.5, 0.5], dtype=float)
    target = np.array([0.25, 0.75], dtype=float)

    ips = compute_ips(rewards, behavior, target)
    snips = compute_snips(rewards, behavior, target)

    assert abs(ips - 0.25) < 1e-6
    assert abs(snips - 0.25) < 1e-6

def test_hand_computed_dr() -> None:
    rewards = np.array([1.0, 0.0], dtype=float)
    behavior = np.array([0.5, 0.5], dtype=float)
    target = np.array([0.25, 0.75], dtype=float)
    q_hat = np.array([0.4, 0.1], dtype=float)

    dr = compute_dr(rewards, behavior, target, q_hat)
    expected = 0.5 * (0.4 + 0.5 * (1.0 - 0.4) + 0.1 + 1.5 * (0.0 - 0.1))
    assert abs(dr - expected) < 1e-6
