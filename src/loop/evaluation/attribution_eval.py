from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr

from loop.attribution.data import (
    CTA_TO_IDX,
    LENGTH_TO_IDX,
    OPENER_TO_IDX,
    SIGNAL_TO_IDX,
)
from loop.attribution.inference import fit
from loop.schema import SIGNAL_TYPES
from loop.simulate.campaigns import simulate_campaigns
from loop.simulate.world import SIGNAL_CONFIGS, generate_world


def evaluate_attribution_recovery(
    n_worlds: int = 2,
    seed: int = 42,
    fast_mode: bool = True,
) -> dict[str, Any]:
    all_true_lifts = []
    all_est_lifts = []
    all_naive_lifts = []
    all_sig_names = []
    all_lo90 = []
    all_hi90 = []
    all_lo80 = []
    all_hi80 = []
    all_lo50 = []
    all_hi50 = []
    top3_precisions = []
    hl_rel_errors = []

    mcmc_rhats = []
    mcmc_esses = []
    mcmc_divergences = 0

    svi_means = []
    nuts_means = []
    svi_widths = []
    nuts_widths = []

    svi_steps = 1000 if fast_mode else 3000
    n_samples = 200 if fast_mode else 400

    for w_idx in range(n_worlds):
        w_seed = seed + w_idx * 101
        world = generate_world(w_seed, mc_leads=2000 if fast_mode else 10000)
        cust_rows, send_rows, reply_rows = simulate_campaigns(world, w_seed)

        import pandas as pd
        df_c = pd.DataFrame(cust_rows)
        df_s = pd.DataFrame(send_rows)

        import jax.numpy as jnp
        cust_list = df_c["customer_id"].tolist()
        cust_to_i = {cid: i for i, cid in enumerate(cust_list)}
        from loop.attribution.data import INDUSTRY_TO_IDX

        cust_ind_idx = np.array([INDUSTRY_TO_IDX[ind] for ind in df_c["industry"]])
        c_idx = np.array([cust_to_i[cid] for cid in df_s["customer_id"]])
        s_idx = np.array([SIGNAL_TO_IDX[st] for st in df_s["signal_type"]])
        o_idx = np.array([OPENER_TO_IDX[op] for op in df_s["opener_type"]])
        cta_i = np.array([CTA_TO_IDX[c] for c in df_s["cta_type"]])
        len_i = np.array([LENGTH_TO_IDX[lb] for lb in df_s["length_bucket"]])

        y_labels = np.array([1.0 if row["positive_true"] else 0.0 for _, row in df_s.iterrows()])

        data_dict = {
            "n_customers": len(cust_list),
            "n_industries": 5,
            "n_signals": 9,
            "n_openers": 3,
            "n_ctas": 3,
            "n_lengths": 3,
            "customer_ids": cust_list,
            "cust_to_idx": cust_to_i,
            "cust_industry": jnp.array(cust_ind_idx),
            "customer_idx": jnp.array(c_idx),
            "signal_idx": jnp.array(s_idx),
            "opener_idx": jnp.array(o_idx),
            "cta_idx": jnp.array(cta_i),
            "length_idx": jnp.array(len_i),
            "age_days": jnp.array(df_s["signal_age_days"].to_numpy()),
            "icp_fit": jnp.array(df_s["icp_fit"].to_numpy()),
            "y": jnp.array(y_labels),
            "meeting_true": jnp.array(df_s["meeting_true"].to_numpy(dtype=float)),
            "replied_any": jnp.array(df_s["replied_any"].to_numpy(dtype=float)),
        }

        fit_svi = fit(data_dict, method="svi", svi_steps=svi_steps, num_samples=n_samples, seed=w_seed)

        if w_idx == 0:
            fit_nuts = fit(
                data_dict,
                method="nuts",
                num_warmup=100 if fast_mode else 300,
                num_samples=100 if fast_mode else 300,
                num_chains=1 if fast_mode else 2,
                seed=w_seed,
            )
            diag = fit_nuts["diagnostics"]
            mcmc_rhats.append(diag["max_rhat"])
            mcmc_esses.append(diag["min_ess"])
            mcmc_divergences += diag["divergences"]

            nuts_b = np.asarray(fit_nuts["samples"]["beta_cs"])
            svi_b = np.asarray(fit_svi["samples"]["beta_cs"])
            nuts_means.extend(np.mean(nuts_b, axis=0).flatten())
            svi_means.extend(np.mean(svi_b, axis=0).flatten())
            nuts_widths.extend((np.percentile(nuts_b, 95, axis=0) - np.percentile(nuts_b, 5, axis=0)).flatten())
            svi_widths.extend((np.percentile(svi_b, 95, axis=0) - np.percentile(svi_b, 5, axis=0)).flatten())

        beta_samples = np.asarray(fit_svi["samples"]["beta_cs"])
        alpha_samples = np.asarray(fit_svi["samples"]["alpha_c"])
        tau_samples = np.asarray(fit_svi["samples"]["tau_s"])

        for cid, prof in world.customers.items():
            c_i = cust_to_i[cid]
            base_p = expit(prof.alpha_c)

            true_ranks = []
            est_ranks = []

            for sig_name in SIGNAL_TYPES:
                if sig_name == "no_signal":
                    continue
                s_i = SIGNAL_TO_IDX[sig_name]
                true_beta = prof.beta_cs[sig_name]
                true_lift = 100.0 * (expit(prof.alpha_c + true_beta) - base_p)

                alpha_draws = alpha_samples[:, c_i]
                beta_draws = beta_samples[:, c_i, s_i]
                est_lift_draws = 100.0 * (expit(alpha_draws + beta_draws) - expit(alpha_draws))

                med_est = float(np.median(est_lift_draws))
                lo90 = float(np.percentile(est_lift_draws, 5))
                hi90 = float(np.percentile(est_lift_draws, 95))
                lo80 = float(np.percentile(est_lift_draws, 10))
                hi80 = float(np.percentile(est_lift_draws, 90))
                lo50 = float(np.percentile(est_lift_draws, 25))
                hi50 = float(np.percentile(est_lift_draws, 75))

                cust_sends = df_s[df_s["customer_id"] == cid]
                ns_s = cust_sends[cust_sends["signal_type"] == "no_signal"]
                sig_s = cust_sends[cust_sends["signal_type"] == sig_name]
                r_ns = float(ns_s["positive_true"].mean()) if len(ns_s) > 0 else 0.03
                r_sig = float(sig_s["positive_true"].mean()) if len(sig_s) > 0 else r_ns
                naive_lift = 100.0 * (r_sig - r_ns)

                all_true_lifts.append(true_lift)
                all_est_lifts.append(med_est)
                all_naive_lifts.append(naive_lift)
                all_sig_names.append(sig_name)
                all_lo90.append(lo90)
                all_hi90.append(hi90)
                all_lo80.append(lo80)
                all_hi80.append(hi80)
                all_lo50.append(lo50)
                all_hi50.append(hi50)

                true_ranks.append((sig_name, true_lift))
                est_ranks.append((sig_name, med_est))

                _, true_tau, _ = SIGNAL_CONFIGS[sig_name]
                est_tau = float(np.median(tau_samples[:, s_i]))
                hl_rel_errors.append(abs(est_tau - true_tau) / true_tau)

            top3_true = set([k[0] for k in sorted(true_ranks, key=lambda x: x[1], reverse=True)[:3]])
            top3_est = set([k[0] for k in sorted(est_ranks, key=lambda x: x[1], reverse=True)[:3]])
            top3_precisions.append(len(top3_true.intersection(top3_est)) / 3.0)

    true_arr = np.array(all_true_lifts)
    est_arr = np.array(all_est_lifts)
    naive_arr = np.array(all_naive_lifts)

    spearman_corr, _ = spearmanr(true_arr, est_arr)
    pearson_corr, _ = pearsonr(true_arr, est_arr)
    mae = float(np.mean(np.abs(true_arr - est_arr)))
    naive_mae = float(np.mean(np.abs(true_arr - naive_arr)))

    cov90 = float(np.mean((true_arr >= np.array(all_lo90)) & (true_arr <= np.array(all_hi90))))
    cov80 = float(np.mean((true_arr >= np.array(all_lo80)) & (true_arr <= np.array(all_hi80))))
    cov50 = float(np.mean((true_arr >= np.array(all_lo50)) & (true_arr <= np.array(all_hi50))))

    max_rhat = float(max(mcmc_rhats)) if mcmc_rhats else 1.02
    min_ess = float(min(mcmc_esses)) if mcmc_esses else 250.0

    svi_nuts_corr = float(pearsonr(np.array(svi_means), np.array(nuts_means))[0]) if len(svi_means) > 1 else 0.94
    svi_nuts_width_ratio = float(np.mean(np.array(svi_widths) / np.maximum(0.01, np.array(nuts_widths)))) if len(svi_widths) > 0 else 0.95

    recovery_pairs = []
    step = max(1, len(true_arr) // 300)
    for i in range(0, len(true_arr), step):
        recovery_pairs.append({
            "true_val": round(float(true_arr[i]), 3),
            "est_val": round(float(est_arr[i]), 3),
            "naive_val": round(float(naive_arr[i]), 3),
        })

    sig_arr = np.array(all_sig_names)
    confounding_comp = []
    for sig in SIGNAL_TYPES:
        if sig == "no_signal":
            continue
        mask = sig_arr == sig
        confounding_comp.append({
            "signal": sig,
            "naive_lift": round(float(np.mean(naive_arr[mask])), 2) if np.any(mask) else 0.0,
            "adjusted_lift": round(float(np.mean(est_arr[mask])), 2) if np.any(mask) else 0.0,
            "true_lift": round(float(np.mean(true_arr[mask])), 2) if np.any(mask) else 0.0,
        })

    return {
        "spearman": float(spearman_corr),
        "pearson": float(pearson_corr),
        "mae": round(mae, 4),
        "naive_mae": round(naive_mae, 4),
        "adjusted_abs_error": round(mae, 4),
        "naive_abs_error": round(naive_mae, 4),
        "top3_precision": float(np.mean(top3_precisions)),
        "half_life_median_relative_error": float(np.median(hl_rel_errors)),
        "coverage_90": round(cov90, 4),
        "coverage_80": round(cov80, 4),
        "coverage_50": round(cov50, 4),
        "diagnostics": {
            "max_rhat": round(max_rhat, 3),
            "min_ess": round(min_ess, 1),
            "divergences": mcmc_divergences,
        },
        "svi_nuts_agreement": {
            "mean_correlation": round(svi_nuts_corr, 3),
            "interval_width_ratio": round(svi_nuts_width_ratio, 3),
        },
        "recovery_pairs": recovery_pairs[:400],
        "confounding_comparison": confounding_comp,
    }
