import json
from pathlib import Path
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
    SIGNAL_TYPES,
    init_db,
)


def compute_posterior_intervals(samples: np.ndarray) -> tuple[float, float, float]:
    med = float(np.median(samples))
    lo = float(np.percentile(samples, 5.0))
    hi = float(np.percentile(samples, 95.0))
    return med, lo, hi

def generate_and_save_summaries(
    fit_res: dict[str, Any],
    db_path: Path,
) -> list[dict[str, Any]]:
    samples = fit_res["samples"]
    data = fit_res["data"]
    customer_ids = data["customer_ids"]

    alpha_c_samples = np.asarray(samples["alpha_c"])
    beta_cs_samples = np.asarray(samples["beta_cs"])
    tau_s_samples = np.asarray(samples["tau_s"])
    a2_c_samples = np.asarray(samples["a2_c"])
    delta_samples = np.asarray(samples["delta"])
    eta_samples = np.asarray(samples["eta"])
    lambda_len_samples = np.asarray(samples["lambda_len"])

    cust_ind_map = np.asarray(data["cust_industry"])
    n_samples = alpha_c_samples.shape[0]

    generic_op_idx = OPENER_TO_IDX["generic_value"]
    default_cta_idx = CTA_TO_IDX["soft_question"]
    default_len_idx = LENGTH_TO_IDX["short"]

    con = init_db(db_path)
    df_sends = con.execute("SELECT customer_id, signal_type, positive_true, signal_age_days, icp_fit FROM sends;").fetchdf()

    summary_rows: list[dict[str, Any]] = []

    for c_idx, cid in enumerate(customer_ids):
        k_idx = int(cust_ind_map[c_idx])
        cust_sends = df_sends[df_sends["customer_id"] == cid]

        no_sig_sends = cust_sends[cust_sends["signal_type"] == "no_signal"]
        base_pos_rate = float(no_sig_sends["positive_true"].mean()) if len(no_sig_sends) > 0 else 0.03

        alpha_c = alpha_c_samples[:, c_idx]
        delta_gen = delta_samples[:, k_idx, generic_op_idx]
        eta_def = eta_samples[:, k_idx, default_cta_idx]
        lam_def = lambda_len_samples[:, default_len_idx]

        base_logit = alpha_c + delta_gen + eta_def + lam_def
        p_base = expit(base_logit)

        for sig_idx, sig in enumerate(SIGNAL_TYPES):
            if sig == "no_signal":
                continue

            beta = beta_cs_samples[:, c_idx, sig_idx]
            p_sig_age0 = expit(base_logit + beta)
            lift_samples = 100.0 * (p_sig_age0 - p_base)
            med_lift, lo_lift, hi_lift = compute_posterior_intervals(lift_samples)

            sig_sends = cust_sends[cust_sends["signal_type"] == sig]
            raw_sig_rate = float(sig_sends["positive_true"].mean()) if len(sig_sends) > 0 else base_pos_rate
            naive_lift = 100.0 * (raw_sig_rate - base_pos_rate)

            tau = tau_s_samples[:, sig_idx]
            hl_samples = tau * np.log(2.0)
            med_hl, lo_hl, hi_hl = compute_posterior_intervals(hl_samples)

            ages_eval = np.linspace(0.0, 90.0, 19)
            decay_curve: list[dict[str, float]] = []
            for age in ages_eval:
                decay_fac = np.exp(-age / tau)
                p_age = expit(base_logit + beta * decay_fac)
                lift_age = 100.0 * (p_age - p_base)
                m, lo, h = compute_posterior_intervals(lift_age)
                decay_curve.append(
                    {
                        "age": float(age),
                        "median": m,
                        "lo90": lo,
                        "hi90": h,
                    }
                )

            extra_json = json.dumps(
                {
                    "naive_lift": round(naive_lift, 4),
                    "half_life_median": round(med_hl, 2),
                    "half_life_lo90": round(lo_hl, 2),
                    "half_life_hi90": round(hi_hl, 2),
                    "decay_curve": decay_curve,
                }
            )

            summary_rows.append(
                {
                    "customer_id": cid,
                    "quantity": "signal_lift_age0",
                    "key": sig,
                    "median": med_lift,
                    "lo90": lo_lift,
                    "hi90": hi_lift,
                    "extra_json": extra_json,
                }
            )

        p_meet_given_pos = expit(a2_c_samples[:, c_idx])

        for arm in ALL_ARMS:
            sig_str, op_str = arm.split(":")
            s_idx = SIGNAL_TO_IDX[sig_str]
            o_idx = OPENER_TO_IDX[op_str]

            arm_sends = cust_sends[cust_sends["signal_type"] == sig_str]
            if len(arm_sends) > 0:
                lead_ages = arm_sends["signal_age_days"].to_numpy(dtype=float)[:64]
                lead_icp = arm_sends["icp_fit"].to_numpy(dtype=float)[:64]
            else:
                lead_ages = np.array([10.0])
                lead_icp = np.array([0.5])

            tau_arm = tau_s_samples[:, s_idx]
            beta_arm = beta_cs_samples[:, c_idx, s_idx]
            delta_arm = delta_samples[:, k_idx, o_idx]

            expected_meets_draws = np.zeros(n_samples, dtype=float)
            for l_age, l_icp in zip(lead_ages, lead_icp, strict=True):
                decay = np.exp(-l_age / tau_arm)
                logits = (
                    alpha_c
                    + 1.0 * (l_icp - 0.5)
                    + beta_arm * decay
                    + delta_arm
                    + eta_def
                    + lam_def
                )
                p_pos_draw = expit(logits)
                expected_meets_draws += p_pos_draw * p_meet_given_pos

            expected_meets_draws = (expected_meets_draws / len(lead_ages)) * 100.0
            med_arm, lo_arm, hi_arm = compute_posterior_intervals(expected_meets_draws)

            summary_rows.append(
                {
                    "customer_id": cid,
                    "quantity": "arm_value",
                    "key": arm,
                    "median": med_arm,
                    "lo90": lo_arm,
                    "hi90": hi_arm,
                    "extra_json": json.dumps({"n_samples": len(lead_ages)}),
                }
            )

    con.execute("DELETE FROM posterior_summaries;")
    import pandas as pd
    df_sum = pd.DataFrame(summary_rows)
    con.register("df_sum_view", df_sum)
    con.execute("INSERT INTO posterior_summaries SELECT * FROM df_sum_view;")
    con.unregister("df_sum_view")
    con.close()

    return summary_rows
