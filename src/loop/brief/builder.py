import json
from typing import Any

import duckdb
import numpy as np

from loop.allocator.policies import policy_loop
from loop.attribution.data import OPENER_TO_IDX, SIGNAL_TO_IDX
from loop.config import DB_PATH
from loop.schema import ALL_ARMS, init_db


def build_customer_brief(
    customer_id: str,
    con: duckdb.DuckDBPyConnection,
    samples: dict[str, Any] | None = None,
    customer_idx: int = 0,
    industry_idx: int = 0,
) -> dict[str, Any]:
    df_cust = con.execute("SELECT * FROM customers WHERE customer_id = ?;", [customer_id]).fetchdf()
    if len(df_cust) == 0:
        raise ValueError(f"Customer {customer_id} not found")
    cust_row = df_cust.iloc[0]

    df_sends = con.execute("SELECT * FROM sends WHERE customer_id = ?;", [customer_id]).fetchdf()
    total_sends = len(df_sends)
    pos_replies = int(df_sends["positive_true"].sum())
    meetings = int(df_sends["meeting_true"].sum())
    weeks = int(df_sends["week"].max()) if total_sends > 0 else 12

    arm_counts = {arm: 0 for arm in ALL_ARMS}
    for arm in df_sends["arm_id"]:
        if arm in arm_counts:
            arm_counts[arm] += 1

    current_shares = {arm: (arm_counts[arm] / total_sends if total_sends > 0 else 1.0 / len(ALL_ARMS)) for arm in ALL_ARMS}

    df_arm_vals = con.execute(
        "SELECT key, median, lo90, hi90 FROM posterior_summaries WHERE customer_id = ? AND quantity = 'arm_value';",
        [customer_id],
    ).fetchdf()

    arm_val_map: dict[str, dict[str, float]] = {}
    if len(df_arm_vals) > 0:
        for _, row in df_arm_vals.iterrows():
            arm_val_map[str(row["key"])] = {
                "median": float(row["median"]),
                "lo90": float(row["lo90"]),
                "hi90": float(row["hi90"]),
            }
    else:
        for arm in ALL_ARMS:
            arm_val_map[arm] = {"median": 1.2, "lo90": 0.5, "hi90": 2.1}

    df_lift = con.execute(
        "SELECT key, median, lo90, hi90, extra_json FROM posterior_summaries WHERE customer_id = ? AND quantity = 'signal_lift_age0';",
        [customer_id],
    ).fetchdf()
    lift_map: dict[str, dict[str, Any]] = {}
    for _, row in df_lift.iterrows():
        extra = json.loads(row["extra_json"]) if row["extra_json"] else {}
        lift_map[str(row["key"])] = {
            "median": float(row["median"]),
            "lo90": float(row["lo90"]),
            "hi90": float(row["hi90"]),
            "extra": extra,
        }

    leads_by_sig: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm in ALL_ARMS:
        sig = arm.split(":")[0]
        sub = df_sends[df_sends["signal_type"] == sig]
        if len(sub) > 0:
            leads_by_sig[sig] = (sub["signal_age_days"].to_numpy()[:64], sub["icp_fit"].to_numpy()[:64])
        else:
            leads_by_sig[sig] = (np.array([10.0]), np.array([0.5]))

    if samples is not None and "alpha_c" in samples:
        rec_probs_arr = policy_loop(
            customer_idx=customer_idx,
            industry_idx=industry_idx,
            samples=samples,
            leads_by_signal=leads_by_sig,
            n_draws=400,
        )
    else:
        sorted_arms = sorted(ALL_ARMS, key=lambda a: arm_val_map[a]["median"], reverse=True)
        rec_probs_arr = np.full(len(ALL_ARMS), 0.05 / len(ALL_ARMS))
        for top_i, top_arm in enumerate(sorted_arms[:3]):
            idx = ALL_ARMS.index(top_arm)
            rec_probs_arr[idx] += 0.95 / 3

    recommended_shares = {ALL_ARMS[i]: float(rec_probs_arr[i]) for i in range(len(ALL_ARMS))}

    n_draws_sim = 1000
    rng_sim = np.random.default_rng(42)
    arm_draws_matrix = np.zeros((len(ALL_ARMS), n_draws_sim), dtype=float)

    from loop.attribution.posterior_store import load_posterior
    saved_post = load_posterior()

    if saved_post is not None and customer_id in saved_post.customer_ids:
        c_i = saved_post.customer_index(customer_id)
        k_i = saved_post.industry_index(customer_id)
        from scipy.special import expit

        from loop.ops.scoring import CustomerContext, LeadArrays, logit_terms
        ctx = CustomerContext(customer_idx=c_i, industry_idx=k_i)
        p_meet = expit(saved_post.draws["a2_c"][:, c_i])
        sample_idx = saved_post.sample_indices(n_draws_sim, rng_sim)

        for a_idx, arm in enumerate(ALL_ARMS):
            s_name, o_name = arm.split(":")
            sub_s = df_sends[df_sends["signal_type"] == s_name]
            age_val = float(sub_s["signal_age_days"].median()) if len(sub_s) > 0 else 10.0
            icp_val = float(sub_s["icp_fit"].median()) if len(sub_s) > 0 else 0.5
            single_lead = LeadArrays(
                signal_idx=np.array([SIGNAL_TO_IDX[s_name]], dtype=int),
                age_days=np.array([age_val], dtype=float),
                icp_fit=np.array([icp_val], dtype=float),
            )
            terms = logit_terms(saved_post, ctx, single_lead, OPENER_TO_IDX[o_name])
            l_sum = terms["base"] + terms["icp"] + terms["signal"] + terms["opener"]
            p_pos = expit(l_sum[:, 0])
            arm_meets = p_pos * p_meet * 100.0
            arm_draws_matrix[a_idx, :] = arm_meets[sample_idx]
    else:
        for a_idx, arm in enumerate(ALL_ARMS):
            info = arm_val_map[arm]
            std = max(0.01, (info["hi90"] - info["lo90"]) / 3.29)
            arm_draws_matrix[a_idx, :] = np.maximum(0.0, rng_sim.normal(info["median"], std, size=n_draws_sim))

    cur_mix_draws = np.zeros(n_draws_sim, dtype=float)
    rec_mix_draws = np.zeros(n_draws_sim, dtype=float)
    for a_idx, arm in enumerate(ALL_ARMS):
        cur_mix_draws += current_shares[arm] * arm_draws_matrix[a_idx, :]
        rec_mix_draws += recommended_shares[arm] * arm_draws_matrix[a_idx, :]

    p_rec_beats_cur = float(np.mean(rec_mix_draws > cur_mix_draws))

    rank_matrix = np.zeros_like(arm_draws_matrix)
    for d in range(n_draws_sim):
        order = np.argsort(-arm_draws_matrix[:, d])
        rank_matrix[order, d] = np.arange(1, len(ALL_ARMS) + 1)

    median_across_arms = np.median(arm_draws_matrix, axis=0)

    cvs = np.array([abs(arm_val_map[a]["hi90"] - arm_val_map[a]["lo90"]) / (2 * max(0.1, arm_val_map[a]["median"])) for a in ALL_ARMS])
    top_quartile_cv = float(np.percentile(cvs, 75.0))

    arm_details: list[dict[str, Any]] = []
    stop_list: list[dict[str, Any]] = []
    test_list: list[dict[str, Any]] = []

    for i, arm in enumerate(ALL_ARMS):
        sig, op = arm.split(":")
        val_info = arm_val_map[arm]
        m = val_info["median"]
        lo = val_info["lo90"]
        h = val_info["hi90"]
        n_s = arm_counts[arm]
        cur_share = current_shares[arm]
        rec_share = recommended_shares[arm]
        cv = cvs[i]

        p_top3 = float(np.mean(rank_matrix[i, :] <= 3))
        p_below_med = float(np.mean(arm_draws_matrix[i, :] < median_across_arms))

        if p_top3 >= 0.50:
            action = "Scale"
            reason = f"Top-performing arm with {m:.2f} expected meetings/100 [{lo:.2f}, {h:.2f}]. Recommended allocation {rec_share*100:.1f}%."
        elif p_below_med >= 0.85:
            action = "Stop"
            reason = f"Underperforming customer median with {m:.2f} meetings/100 [{lo:.2f}, {h:.2f}] over {n_s} sends."
            stop_list.append({"arm_id": arm, "action": action, "reason": reason, "expected_meetings": m})
        elif cv >= top_quartile_cv and n_s < 100:
            action = "Test"
            reason = f"High posterior uncertainty (CV {cv:.2f}) with only {n_s} sends observed. Test to resolve value."
            test_list.append({"arm_id": arm, "action": action, "reason": reason, "expected_meetings": m})
        else:
            action = "Hold"
            reason = f"Stable performance ({m:.2f} meetings/100) within holding band across {n_s} sends."

        arm_details.append(
            {
                "arm_id": arm,
                "signal_type": sig,
                "opener_type": op,
                "current_share": round(cur_share, 4),
                "recommended_share": round(rec_share, 4),
                "expected_meetings_median": round(m, 2),
                "expected_meetings_lo90": round(lo, 2),
                "expected_meetings_hi90": round(h, 2),
                "n_sends": n_s,
                "action": action,
                "reason": reason,
            }
        )

    cur_exp = float(np.median(cur_mix_draws))
    cur_lo = float(np.percentile(cur_mix_draws, 5.0))
    cur_hi = float(np.percentile(cur_mix_draws, 95.0))

    rec_exp = float(np.median(rec_mix_draws))
    rec_lo = float(np.percentile(rec_mix_draws, 5.0))
    rec_hi = float(np.percentile(rec_mix_draws, 95.0))

    rel_gain = ((rec_exp - cur_exp) / cur_exp * 100.0) if cur_exp > 0 else 0.0

    brief_data = {
        "customer_id": customer_id,
        "customer_name": cust_row["name"],
        "industry": cust_row["industry"],
        "primary_language": cust_row["primary_language"],
        "window": {
            "weeks": weeks,
            "total_sends": total_sends,
            "positive_replies": pos_replies,
            "meetings": meetings,
        },
        "kpi_strip": {
            "current_expected_meetings": round(cur_exp, 2),
            "current_lo90": round(cur_lo, 2),
            "current_hi90": round(cur_hi, 2),
            "recommended_expected_meetings": round(rec_exp, 2),
            "recommended_lo90": round(rec_lo, 2),
            "recommended_hi90": round(rec_hi, 2),
            "relative_gain_pct": round(rel_gain, 1),
            "p_recommended_beats_current": round(p_rec_beats_cur, 2),
        },
        "allocation_table": arm_details,
        "stop_list": stop_list,
        "test_list": test_list,
    }

    return brief_data

def generate_all_briefs() -> None:
    con = init_db(DB_PATH)
    customers = con.execute("SELECT customer_id FROM customers ORDER BY customer_id;").fetchall()

    for row in customers:
        cid = str(row[0])
        brief = build_customer_brief(cid, con)
        brief_json = json.dumps(brief)
        con.execute(
            "INSERT OR REPLACE INTO briefs (customer_id, brief_json) VALUES (?, ?);",
            [cid, brief_json],
        )
    con.close()

if __name__ == "__main__":
    generate_all_briefs()
