from collections.abc import Callable
from typing import Any

import numpy as np


def bootstrap_ci(
    data: list[Any] | np.ndarray,
    stat_fn: Callable[[list[Any] | np.ndarray], float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(data)
    arr = np.asarray(data)
    point_est = float(stat_fn(arr))

    boot_stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        sample = arr[indices]
        boot_stats[i] = stat_fn(sample)

    lower = float(np.percentile(boot_stats, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point_est, lower, upper

def paired_bootstrap_difference(
    data_a: np.ndarray,
    data_b: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(data_a)
    diffs = data_a - data_b
    mean_diff = float(np.mean(diffs))

    boot_diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        boot_diffs[i] = float(np.mean(diffs[indices]))

    lower = float(np.percentile(boot_diffs, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    return mean_diff, lower, upper
