from typing import Any

import numpy as np

from loop.evaluation.bootstrap import bootstrap_ci
from loop.evaluation.metrics import (
    compute_dr,
    compute_effective_sample_size,
    compute_ips,
    compute_snips,
)


def evaluate_ope(
    rewards: np.ndarray,
    behavior_propensities: np.ndarray,
    target_propensities: np.ndarray,
    q_model_rewards: np.ndarray | None = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    n_samples = len(rewards)
    if q_model_rewards is None:
        q_model_rewards = np.full(n_samples, float(np.mean(rewards)))

    ips_val = compute_ips(rewards, behavior_propensities, target_propensities)
    snips_val = compute_snips(rewards, behavior_propensities, target_propensities)
    dr_val = compute_dr(rewards, behavior_propensities, target_propensities, q_model_rewards)
    ess = compute_effective_sample_size(behavior_propensities, target_propensities)

    def ips_stat(idx_arr: np.ndarray) -> float:
        return compute_ips(rewards[idx_arr], behavior_propensities[idx_arr], target_propensities[idx_arr])

    def snips_stat(idx_arr: np.ndarray) -> float:
        return compute_snips(rewards[idx_arr], behavior_propensities[idx_arr], target_propensities[idx_arr])

    def dr_stat(idx_arr: np.ndarray) -> float:
        return compute_dr(rewards[idx_arr], behavior_propensities[idx_arr], target_propensities[idx_arr], q_model_rewards[idx_arr])

    indices = np.arange(n_samples)
    _, ips_lo, ips_hi = bootstrap_ci(indices, ips_stat, n_resamples=n_resamples, seed=seed)
    _, snips_lo, snips_hi = bootstrap_ci(indices, snips_stat, n_resamples=n_resamples, seed=seed + 1)
    _, dr_lo, dr_hi = bootstrap_ci(indices, dr_stat, n_resamples=n_resamples, seed=seed + 2)

    return {
        "ips": {
            "value": round(ips_val, 6),
            "lo95": round(ips_lo, 6),
            "hi95": round(ips_hi, 6),
        },
        "snips": {
            "value": round(snips_val, 6),
            "lo95": round(snips_lo, 6),
            "hi95": round(snips_hi, 6),
        },
        "dr": {
            "value": round(dr_val, 6),
            "lo95": round(dr_lo, 6),
            "hi95": round(dr_hi, 6),
        },
        "effective_sample_size": round(ess, 2),
        "n_samples": n_samples,
    }
