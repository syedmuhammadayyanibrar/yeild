from typing import Any

import numpy as np

from loop.allocator.ope import evaluate_ope
from loop.schema import ALL_ARMS


def evaluate_ope_simulation(
    seed: int = 42,
    fast_mode: bool = True,
) -> dict[str, Any]:
    n_seeds = 10 if fast_mode else 50
    n_arms = len(ALL_ARMS)
    n_sends_per_seed = 1000

    true_arm_means = np.linspace(0.008, 0.035, n_arms)

    behavior_policy = np.full(n_arms, 1.0 / n_arms)

    target_policy = np.full(n_arms, 0.05 / n_arms)
    target_policy[0] += 0.35
    target_policy[1] += 0.35
    target_policy[2] += 0.25

    ground_truth_target_val = float(np.sum(target_policy * true_arm_means))

    dr_biases = []
    ips_biases = []
    snips_biases = []
    dr_errors_sq = []
    esses = []
    dr_covered = 0

    for s_i in range(n_seeds):
        rng = np.random.default_rng(seed + s_i * 31)

        arms_sampled = rng.choice(n_arms, size=n_sends_per_seed, p=behavior_policy)
        rewards = np.zeros(n_sends_per_seed, dtype=float)
        for i in range(n_sends_per_seed):
            arm = arms_sampled[i]
            p_meet = true_arm_means[arm]
            rewards[i] = 1.0 if rng.random() < p_meet else 0.0

        b_props = np.array([behavior_policy[a] for a in arms_sampled])
        t_props = np.array([target_policy[a] for a in arms_sampled])

        q_hat = np.array([true_arm_means[a] + rng.normal(0, 0.003) for a in arms_sampled])

        ope_res = evaluate_ope(rewards, b_props, t_props, q_model_rewards=q_hat, n_resamples=500, seed=seed)

        dr_est = ope_res["dr"]["value"]
        ips_est = ope_res["ips"]["value"]
        snips_est = ope_res["snips"]["value"]
        ess = ope_res["effective_sample_size"]

        dr_bias = (dr_est - ground_truth_target_val) / ground_truth_target_val
        ips_bias = (ips_est - ground_truth_target_val) / ground_truth_target_val
        snips_bias = (snips_est - ground_truth_target_val) / ground_truth_target_val

        dr_biases.append(dr_bias)
        ips_biases.append(ips_bias)
        snips_biases.append(snips_bias)
        dr_errors_sq.append((dr_est - ground_truth_target_val) ** 2)
        esses.append(ess)

        if ope_res["dr"]["lo95"] <= ground_truth_target_val <= ope_res["dr"]["hi95"]:
            dr_covered += 1

    mean_dr_bias = float(np.mean(np.abs(dr_biases))) * 100.0
    mean_ips_bias = float(np.mean(np.abs(ips_biases))) * 100.0
    mean_snips_bias = float(np.mean(np.abs(snips_biases))) * 100.0
    dr_rmse = float(np.sqrt(np.mean(dr_errors_sq)))
    coverage = float(dr_covered / n_seeds)

    return {
        "ground_truth_target_value": round(ground_truth_target_val, 5),
        "dr_abs_bias_pct": round(mean_dr_bias, 2),
        "ips_abs_bias_pct": round(mean_ips_bias, 2),
        "snips_abs_bias_pct": round(mean_snips_bias, 2),
        "dr_rmse": round(dr_rmse, 5),
        "coverage_95": round(coverage, 2),
        "effective_sample_size": round(float(np.mean(esses)), 1),
    }
