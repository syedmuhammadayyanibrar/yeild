import json
from typing import Any

import duckdb
import numpy as np
from scipy.special import expit

from loop.config import DB_PATH, REPORTS_DIR
from loop.ops.leads import ensure_lead_pool_for_customer
from loop.ops.plan import fetch_operator_plan
from loop.schema import ALL_ARMS, init_db
from loop.simulate.campaigns import compute_historical_arm_propensities
from loop.simulate.replies import SPLIT_CORE_IDS, compose_reply_text
from loop.simulate.world import (
    compute_single_send_positive_prob,
    generate_world,
)


def init_ledger_for_customer(customer_id: str, con: duckdb.DuckDBPyConnection) -> None:
    count = con.execute("SELECT COUNT(*) FROM weekly_ledger WHERE customer_id = ?;", [customer_id]).fetchone()[0]
    if count > 0:
        return

    df_sends = con.execute(
        """
        SELECT week, COUNT(*) as n_sends, SUM(CASE WHEN positive_true THEN 1 ELSE 0 END) as pos_rep,
               SUM(CASE WHEN meeting_true THEN 1 ELSE 0 END) as meets
        FROM sends
        WHERE customer_id = ?
        GROUP BY week
        ORDER BY week;
        """,
        [customer_id],
    ).fetchdf()

    ledger_rows = []
    for _, row in df_sends.iterrows():
        w = int(row["week"])
        s_count = int(row["n_sends"])
        pos = int(row["pos_rep"])
        meets = int(row["meets"])
        ledger_rows.append((w, customer_id, "Historical", s_count, pos, meets, float(meets), 0.0))

    if ledger_rows:
        con.executemany(
            """
            INSERT OR REPLACE INTO weekly_ledger (
                week, customer_id, policy, sends, positive_replies, meetings,
                counterfactual_meetings, difference_meetings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            ledger_rows,
        )


def fetch_ledger_data(customer_id: str) -> dict[str, Any]:
    con = init_db(DB_PATH)
    init_ledger_for_customer(customer_id, con)

    df_led = con.execute(
        "SELECT * FROM weekly_ledger WHERE customer_id = ? ORDER BY week ASC;",
        [customer_id]
    ).fetchdf()
    con.close()

    rows = []
    total_loop_meets = 0
    total_cf_meets = 0.0

    for _, r in df_led.iterrows():
        meets = int(r["meetings"])
        cf = float(r["counterfactual_meetings"])
        w = int(r["week"])
        if w > 12:
            total_loop_meets += meets
            total_cf_meets += cf

        rows.append({
            "week": w,
            "plan": str(r["policy"]),
            "sends": int(r["sends"]),
            "positive_replies": int(r["positive_replies"]),
            "meetings": meets,
            "counterfactual_meetings": round(cf, 1),
            "difference_meetings": round(float(r["difference_meetings"]), 1),
        })

    gates_list = []
    eval_file = REPORTS_DIR / "eval_results.json"
    if eval_file.exists():
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                edata = json.load(f)
            for g in edata.get("gates", []):
                val_str = str(g.get("value", ""))
                gates_list.append({
                    "gate_id": str(g.get("id")),
                    "description": str(g.get("description")),
                    "status": str(g.get("status")),
                    "value": val_str[:30],
                })
        except Exception:
            pass

    return {
        "customer_id": customer_id,
        "rows": rows,
        "total_meetings_loop": total_loop_meets,
        "total_meetings_counterfactual": round(total_cf_meets, 1),
        "net_gain_meetings": round(total_loop_meets - total_cf_meets, 1),
        "gates": gates_list,
    }


def execute_next_week_simulation(customer_id: str) -> dict[str, Any]:
    con = init_db(DB_PATH)
    init_ledger_for_customer(customer_id, con)
    ensure_lead_pool_for_customer(customer_id, con)

    max_w = con.execute("SELECT MAX(week) FROM weekly_ledger WHERE customer_id = ?;", [customer_id]).fetchone()[0]
    next_week = (max_w or 12) + 1

    df_leads = con.execute(
        "SELECT * FROM lead_pool WHERE customer_id = ? AND is_sent = false LIMIT 60;",
        [customer_id]
    ).fetchdf()

    if len(df_leads) < 20:
        ensure_lead_pool_for_customer(customer_id, con)
        df_leads = con.execute(
            "SELECT * FROM lead_pool WHERE customer_id = ? AND is_sent = false LIMIT 60;",
            [customer_id]
        ).fetchdf()

    plan = fetch_operator_plan(customer_id)
    rec_shares = {item["arm_id"]: item["recommended_share_pct"] / 100.0 for item in plan["recommendations"]}
    arms_pool = list(ALL_ARMS)
    probs = np.array([rec_shares.get(a, 0.0) for a in arms_pool], dtype=float)
    if probs.sum() <= 0:
        probs = np.full(len(arms_pool), 1.0 / len(arms_pool))
    else:
        probs = probs / probs.sum()

    world = generate_world(seed=7 + next_week * 13)
    profile = world.customers.get(customer_id)
    if profile is None:
        profile = next(iter(world.customers.values()))

    hist_props = compute_historical_arm_propensities(profile)
    hist_probs = np.array([hist_props[a] for a in arms_pool], dtype=float)
    hist_probs = hist_probs / hist_probs.sum()

    rng = np.random.default_rng(700 + next_week * 31)
    p_meet_given_pos = float(expit(profile.meeting_logit))

    loop_pos_count = 0
    loop_meet_count = 0
    cf_meet_count = 0.0

    send_inserts = []
    reply_inserts = []
    lead_updates = []
    train_cores = SPLIT_CORE_IDS["train"]

    for i, row in df_leads.iterrows():
        lead_id = str(row["lead_id"])
        sig = str(row["signal_type"])
        age_days = float(row["signal_age_days"])
        icp_fit = float(row["icp_fit"])

        best_arm_idx = int(rng.choice(len(arms_pool), p=probs))
        chosen_arm = arms_pool[best_arm_idx]
        chosen_op = chosen_arm.split(":")[1]

        p_pos = compute_single_send_positive_prob(
            profile, sig, age_days, icp_fit, chosen_op, profile.default_cta, profile.default_length
        )
        is_pos = bool(rng.random() < p_pos)
        is_meet = bool(rng.random() < (p_meet_given_pos if is_pos else 0.0))

        if is_pos:
            loop_pos_count += 1
        if is_meet:
            loop_meet_count += 1

        cf_arm_idx = int(rng.choice(len(arms_pool), p=hist_probs))
        cf_arm = arms_pool[cf_arm_idx]
        cf_op = cf_arm.split(":")[1]
        cf_p_pos = compute_single_send_positive_prob(
            profile, sig, age_days, icp_fit, cf_op, profile.default_cta, profile.default_length
        )
        cf_meet_count += cf_p_pos * p_meet_given_pos

        sid = f"send_w{next_week}_{i:04d}_{customer_id}"
        send_inserts.append((
            sid, customer_id, next_week, sig, age_days, icp_fit, chosen_op,
            profile.default_cta, profile.default_length, profile.primary_language,
            chosen_arm, f"loop_v{next_week-12}", float(probs[best_arm_idx]),
            is_pos or (rng.random() < profile.p_non_positive_reply), is_pos, is_meet
        ))
        lead_updates.append((chosen_arm, sid, lead_id))

        if is_pos:
            rep_class = "interested" if rng.random() < 0.60 else "info_request"
            core_id = int(rng.choice(train_cores))
            rep_text, _ = compose_reply_text(rep_class, profile.primary_language, core_id, rng)
            rid = f"rep_w{next_week}_{i:04d}_{customer_id}"
            reply_inserts.append((
                rid, sid, customer_id, f"2026-10-{next_week:02d}T10:00:00Z", rep_text,
                rep_class, profile.primary_language, None, None, None, None, "train"
            ))

    n_sends = len(df_leads)
    cf_meets_round = round(cf_meet_count, 1)
    diff = round(loop_meet_count - cf_meets_round, 1)
    policy_name = f"Yeild v{next_week - 12}"

    con.execute(
        """
        INSERT OR REPLACE INTO weekly_ledger (
            week, customer_id, policy, sends, positive_replies, meetings,
            counterfactual_meetings, difference_meetings
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        [next_week, customer_id, policy_name, n_sends, loop_pos_count, loop_meet_count, cf_meets_round, diff],
    )

    if send_inserts:
        con.executemany(
            """
            INSERT OR REPLACE INTO sends (
                send_id, customer_id, week, signal_type, signal_age_days, icp_fit,
                opener_type, cta_type, length_bucket, language, arm_id, policy,
                propensity, replied_any, positive_true, meeting_true
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            send_inserts,
        )

    if reply_inserts:
        con.executemany(
            """
            INSERT OR REPLACE INTO replies (
                reply_id, send_id, customer_id, received_at, text, true_class,
                true_language, referral_name, referral_email, revisit_window_start,
                revisit_window_end, split
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            reply_inserts,
        )

    if lead_updates:
        con.executemany(
            "UPDATE lead_pool SET is_sent = true, allocated_arm = ?, send_id = ? WHERE lead_id = ?;",
            lead_updates,
        )

    con.close()

    new_row = {
        "week": next_week,
        "plan": policy_name,
        "sends": n_sends,
        "positive_replies": loop_pos_count,
        "meetings": loop_meet_count,
        "counterfactual_meetings": cf_meets_round,
        "difference_meetings": diff,
    }

    return {
        "success": True,
        "new_row": new_row,
        "message": f"Successfully executed Week {next_week} with {loop_meet_count} booked meetings (+{diff} over counterfactual).",
    }
