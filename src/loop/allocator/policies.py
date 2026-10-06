from typing import Any

import numpy as np
from scipy.special import expit

from loop.attribution.data import (
    CTA_TO_IDX,
    LENGTH_TO_IDX,
    OPENER_TO_IDX,
    SIGNAL_TO_IDX,
)
from loop.schema import (
    ALL_ARMS,
)


def policy_uniform() -> np.ndarray:
    n_arms = len(ALL_ARMS)
    return np.full(n_arms, 1.0 / n_arms, dtype=float)

def policy_historical_static(
    propensities_map: dict[str, float],
) -> np.ndarray:
    vec = np.array([propensities_map[a] for a in ALL_ARMS], dtype=float)
    return vec / vec.sum()

def policy_epsilon_greedy(
    arm_counts: np.ndarray,
    arm_meetings: np.ndarray,
    global_meeting_rate: float = 0.015,
    epsilon: float = 0.10,
) -> np.ndarray:
    n_arms = len(ALL_ARMS)
    empirical_rates = np.zeros(n_arms, dtype=float)

    for i in range(n_arms):
        if arm_counts[i] > 0:
            empirical_rates[i] = arm_meetings[i] / arm_counts[i]
        else:
            empirical_rates[i] = global_meeting_rate

    best_arm = int(np.argmax(empirical_rates))
    probs = np.full(n_arms, epsilon / n_arms, dtype=float)
    probs[best_arm] += 1.0 - epsilon
    return probs

def policy_thompson_flat(
    arm_counts: np.ndarray,
    arm_meetings: np.ndarray,
    global_meeting_rate: float = 0.015,
    prior_strength: float = 50.0,
    n_draws: int = 400,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)

    n_arms = len(ALL_ARMS)
    prior_a = global_meeting_rate * prior_strength
    prior_b = (1.0 - global_meeting_rate) * prior_strength

    wins = np.zeros(n_arms, dtype=float)
    for _ in range(n_draws):
        sampled_rates = np.zeros(n_arms, dtype=float)
        for i in range(n_arms):
            a = prior_a + arm_meetings[i]
            b = prior_b + (arm_counts[i] - arm_meetings[i])
            sampled_rates[i] = rng.beta(max(0.1, a), max(0.1, b))
        best_i = int(np.argmax(sampled_rates))
        wins[best_i] += 1.0

    raw_probs = wins / n_draws
    return 0.95 * raw_probs + 0.05 * (1.0 / n_arms)

def policy_loop(
    customer_idx: int,
    industry_idx: int,
    samples: dict[str, Any],
    leads_by_signal: dict[str, tuple[np.ndarray, np.ndarray]],
    n_draws: int = 400,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    n_arms = len(ALL_ARMS)

    alpha_c_all = np.asarray(samples["alpha_c"])
    beta_cs_all = np.asarray(samples["beta_cs"])
    tau_s_all = np.asarray(samples["tau_s"])
    a2_c_all = np.asarray(samples["a2_c"])
    delta_all = np.asarray(samples["delta"])
    eta_all = np.asarray(samples["eta"])
    lambda_len_all = np.asarray(samples["lambda_len"])

    available_samples = alpha_c_all.shape[0]
    draw_indices = np.linspace(0, available_samples - 1, n_draws, dtype=int)

    default_cta_idx = CTA_TO_IDX["soft_question"]
    default_len_idx = LENGTH_TO_IDX["short"]

    arm_signals = [a.split(":")[0] for a in ALL_ARMS]
    arm_openers = [a.split(":")[1] for a in ALL_ARMS]

    sig_indices = [SIGNAL_TO_IDX[s] for s in arm_signals]
    op_indices = [OPENER_TO_IDX[o] for o in arm_openers]

    wins = np.zeros(n_arms, dtype=float)

    for d_idx in draw_indices:
        alpha = alpha_c_all[d_idx, customer_idx]
        p_meet = expit(a2_c_all[d_idx, customer_idx])
        eta = eta_all[d_idx, industry_idx, default_cta_idx]
        lam = lambda_len_all[d_idx, default_len_idx]

        arm_expected_meets = np.zeros(n_arms, dtype=float)

        for a_i in range(n_arms):
            sig_str = arm_signals[a_i]
            s_i = sig_indices[a_i]
            o_i = op_indices[a_i]

            beta = beta_cs_all[d_idx, customer_idx, s_i]
            tau = tau_s_all[d_idx, s_i]
            delta = delta_all[d_idx, industry_idx, o_i]

            ages, icp = leads_by_signal.get(sig_str, (np.array([10.0]), np.array([0.5])))
            ages_sub = ages[:64]
            icp_sub = icp[:64]

            decays = np.exp(-ages_sub / tau)
            logits = alpha + 1.0 * (icp_sub - 0.5) + beta * decays + delta + eta + lam
            p_pos = expit(logits)
            arm_expected_meets[a_i] = float(np.mean(p_pos) * p_meet)

        best_arm = int(np.argmax(arm_expected_meets))
        wins[best_arm] += 1.0

    raw_probs = wins / n_draws
    return 0.95 * raw_probs + 0.05 * (1.0 / n_arms)

def policy_oracle(
    ground_truth_arm_values: dict[str, float],
) -> np.ndarray:
    n_arms = len(ALL_ARMS)
    values = np.array([ground_truth_arm_values[a] for a in ALL_ARMS], dtype=float)
    best_arm = int(np.argmax(values))
    probs = np.zeros(n_arms, dtype=float)
    probs[best_arm] = 1.0
    return probs
