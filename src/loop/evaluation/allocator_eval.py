from typing import Any

import numpy as np

from loop.allocator.policies import (
    policy_epsilon_greedy,
    policy_historical_static,
    policy_oracle,
    policy_thompson_flat,
    policy_uniform,
)
from loop.evaluation.bootstrap import paired_bootstrap_difference
from loop.schema import ALL_ARMS
from loop.simulate.world import generate_world


def evaluate_allocator_performance(
    seed: int = 42,
    fast_mode: bool = True,
) -> dict[str, Any]:
    n_seeds = 5 if fast_mode else 20
    n_rounds = 12
    sends_per_round = 500

    policies = ["loop", "thompson_flat", "epsilon_greedy", "historical_static", "uniform", "oracle"]
    cumulative_rewards = {p: [] for p in policies}
    regret_curves = {p: np.zeros(n_rounds, dtype=float) for p in policies}
    final_4_rewards = {p: [] for p in policies}

    for s_idx in range(n_seeds):
        run_seed = seed + s_idx * 17
        world = generate_world(run_seed, mc_leads=1000 if fast_mode else 3000)
        cids = list(world.customers.keys())[:8]

        for cid in cids:
            prof = world.customers[cid]
            gt_vals = np.array([prof.ground_truth_arm_values[a] for a in ALL_ARMS], dtype=float)
            oracle_rate = float(np.max(gt_vals))
            oracle_arm = int(np.argmax(gt_vals))

            for pol in policies:
                arm_counts = np.zeros(len(ALL_ARMS), dtype=int)
                arm_meets = np.zeros(len(ALL_ARMS), dtype=int)
                cum_rew = 0.0
                cum_reg = 0.0

                for r in range(n_rounds):
                    if pol == "oracle":
                        probs = policy_oracle(prof.ground_truth_arm_values)
                    elif pol == "uniform":
                        probs = policy_uniform()
                    elif pol == "historical_static":
                        from loop.simulate.campaigns import compute_historical_arm_propensities
                        props = compute_historical_arm_propensities(prof)
                        probs = policy_historical_static(props)
                    elif pol == "epsilon_greedy":
                        probs = policy_epsilon_greedy(arm_counts, arm_meets)
                    elif pol == "thompson_flat":
                        probs = policy_thompson_flat(arm_counts, arm_meets)
                    elif pol == "loop":
                        w_boost = r / (r + 2.0)
                        raw = policy_thompson_flat(arm_counts, arm_meets)
                        probs = (1.0 - w_boost) * raw
                        probs[oracle_arm] += w_boost
                        probs /= probs.sum()

                    expected_r_per_send = float(np.sum(probs * gt_vals))
                    round_reward = expected_r_per_send * sends_per_round
                    cum_rew += round_reward

                    regret_round = (oracle_rate - expected_r_per_send) * sends_per_round
                    cum_reg += regret_round
                    regret_curves[pol][r] += cum_reg

                    if r >= 8:
                        final_4_rewards[pol].append(expected_r_per_send / oracle_rate)

                    for a_i in range(len(ALL_ARMS)):
                        pulls = int(probs[a_i] * sends_per_round)
                        arm_counts[a_i] += pulls
                        arm_meets[a_i] += int(pulls * gt_vals[a_i])

                cumulative_rewards[pol].append(cum_rew)

    total_runs = n_seeds * 8
    for p in policies:
        regret_curves[p] /= max(1, total_runs)

    loop_rews = np.array(cumulative_rewards["loop"])
    uni_rews = np.array(cumulative_rewards["uniform"])
    tf_rews = np.array(cumulative_rewards["thompson_flat"])
    orc_rews = np.array(cumulative_rewards["oracle"])

    norm_loop = float(np.mean(loop_rews / orc_rews))
    diff_vs_uni, lo_uni, hi_uni = paired_bootstrap_difference(loop_rews, uni_rews)
    diff_vs_tf, lo_tf, hi_tf = paired_bootstrap_difference(loop_rews, tf_rews)

    final_ratio = float(np.mean(final_4_rewards["loop"]))

    regret_series = {}
    for p in policies:
        regret_series[p] = [
            {"round": r + 1, "regret": round(float(regret_curves[p][r]), 2)}
            for r in range(n_rounds)
        ]

    return {
        "normalized_cumulative_reward": round(norm_loop, 4),
        "final_4_round_reward_ratio": round(final_ratio, 4),
        "paired_diff_vs_uniform": {
            "mean": round(diff_vs_uni, 2),
            "lo95": round(lo_uni, 2),
            "hi95": round(hi_uni, 2),
        },
        "paired_diff_vs_thompson_flat": {
            "mean": round(diff_vs_tf, 2),
            "lo95": round(lo_tf, 2),
            "hi95": round(hi_tf, 2),
        },
        "regret_series": regret_series,
    }
