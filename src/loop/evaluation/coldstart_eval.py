from typing import Any

import numpy as np
from scipy.special import expit

from loop.simulate.campaigns import simulate_campaigns
from loop.simulate.world import generate_world


def evaluate_cold_start(
    seed: int = 42,
    fast_mode: bool = True,
) -> dict[str, Any]:
    world = generate_world(seed, mc_leads=1000 if fast_mode else 5000)
    _, send_rows, _ = simulate_campaigns(world, seed)

    import pandas as pd
    df_sends = pd.DataFrame(send_rows)

    qualifying_cids = []
    for ind in ["saas_devtools", "ecommerce_dtc", "fintech", "marketing_agencies", "industrial_b2b"]:
        c_in_ind = [c for c, p in world.customers.items() if p.industry == ind and len(df_sends[df_sends["customer_id"] == c]) >= 800]
        qualifying_cids.extend(c_in_ind[: (1 if fast_mode else 2)])

    k_values = [0, 100, 300] if fast_mode else [0, 25, 100, 300, 800]

    losses_ind = {k: [] for k in k_values}
    losses_glob = {k: [] for k in k_values}
    losses_flat = {k: [] for k in k_values}

    for cid in qualifying_cids:
        prof = world.customers[cid]
        c_sends = df_sends[df_sends["customer_id"] == cid]
        eval_sends = c_sends.iloc[min(len(c_sends), 800):] if len(c_sends) > 800 else c_sends.iloc[len(c_sends)//2:]
        if len(eval_sends) == 0:
            continue

        true_eval_y = eval_sends["positive_true"].to_numpy().astype(float)

        ind_base = float(expit(world.alpha_k[prof.industry]))
        glob_base = float(expit(float(np.mean(list(world.alpha_k.values())))))

        for k in k_values:
            k_sends = c_sends.iloc[:k]
            obs_pos = float(k_sends["positive_true"].sum()) if k > 0 else 0.0

            w_k = k / (k + 150.0)
            p_ind = (1.0 - w_k) * ind_base + w_k * (obs_pos / max(1, k))
            p_glob = (1.0 - w_k) * glob_base + w_k * (obs_pos / max(1, k))
            p_flat = 0.03 if k == 0 else (obs_pos / k)

            eps = 1e-12
            p_ind_c = np.clip(p_ind, eps, 1.0 - eps)
            p_glob_c = np.clip(p_glob, eps, 1.0 - eps)
            p_flat_c = np.clip(p_flat, eps, 1.0 - eps)

            loss_i = -float(np.mean(true_eval_y * np.log(p_ind_c) + (1.0 - true_eval_y) * np.log(1.0 - p_ind_c)))
            loss_g = -float(np.mean(true_eval_y * np.log(p_glob_c) + (1.0 - true_eval_y) * np.log(1.0 - p_glob_c)))
            loss_f = -float(np.mean(true_eval_y * np.log(p_flat_c) + (1.0 - true_eval_y) * np.log(1.0 - p_flat_c)))

            losses_ind[k].append(loss_i)
            losses_glob[k].append(loss_g)
            losses_flat[k].append(loss_f)

    curves = {
        "industry_pooled": [
            {
                "k": k,
                "log_loss": round(float(np.mean(losses_ind[k])), 4),
                "lo95": round(float(np.percentile(losses_ind[k], 5)), 4),
                "hi95": round(float(np.percentile(losses_ind[k], 95)), 4),
            }
            for k in k_values
        ],
        "global_pooled": [
            {
                "k": k,
                "log_loss": round(float(np.mean(losses_glob[k])), 4),
                "lo95": round(float(np.percentile(losses_glob[k], 5)), 4),
                "hi95": round(float(np.percentile(losses_glob[k], 95)), 4),
            }
            for k in k_values
        ],
        "flat": [
            {
                "k": k,
                "log_loss": round(float(np.mean(losses_flat[k])), 4),
                "lo95": round(float(np.percentile(losses_flat[k], 5)), 4),
                "hi95": round(float(np.percentile(losses_flat[k], 95)), 4),
            }
            for k in k_values
        ],
    }

    gain_k0 = float((curves["flat"][0]["log_loss"] - curves["industry_pooled"][0]["log_loss"]) / curves["flat"][0]["log_loss"]) * 100.0

    return {
        "curves": curves,
        "k0_relative_gain_pct": round(gain_k0, 2),
        "holdouts_count": len(qualifying_cids),
    }
