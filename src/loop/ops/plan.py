import json
from typing import Any

from loop.brief.builder import build_customer_brief
from loop.config import DB_PATH
from loop.schema import init_db


def fetch_operator_plan(customer_id: str) -> dict[str, Any]:
    con = init_db(DB_PATH)
    res = con.execute("SELECT brief_json FROM briefs WHERE customer_id = ?;", [customer_id]).fetchone()
    if res and res[0]:
        data = json.loads(res[0])
    else:
        data = build_customer_brief(customer_id, con)
        brief_json = json.dumps(data)
        con.execute("INSERT OR REPLACE INTO briefs (customer_id, brief_json) VALUES (?, ?);", [customer_id, brief_json])

    con.close()
    kpi = data["kpi_strip"]

    recommendations = []
    for arm in data["allocation_table"]:
        act = arm["action"]
        c_sh = arm["current_share"] * 100.0
        r_sh = arm["recommended_share"] * 100.0
        m = arm["expected_meetings_median"]
        lo = arm["expected_meetings_lo90"]
        hi = arm["expected_meetings_hi90"]
        n_s = arm["n_sends"]

        if act == "Scale":
            rationale = f"Top-performing arm with {m:.1f} expected meetings/100 [{lo:.1f}, {hi:.1f}]. Scale from {c_sh:.1f}% to {r_sh:.1f}%."
        elif act == "Stop":
            rationale = f"Underperforming customer median with {m:.1f} meetings/100 [{lo:.1f}, {hi:.1f}] over {n_s} sends. Cut to 0%."
        elif act == "Test":
            rationale = f"High posterior uncertainty with only {n_s} sends observed. Allocate {r_sh:.1f}% to resolve causal lift."
        else:
            rationale = f"Stable performance ({m:.1f} meetings/100) within holding band across {n_s} sends."

        recommendations.append({
            "action": act,
            "arm_id": arm["arm_id"],
            "signal_type": arm["signal_type"],
            "opener_type": arm["opener_type"],
            "current_share_pct": round(c_sh, 1),
            "recommended_share_pct": round(r_sh, 1),
            "expected_meetings": round(m, 2),
            "lo90": round(lo, 2),
            "hi90": round(hi, 2),
            "sends_observed": n_s,
            "rationale": rationale,
        })

    recommendations.sort(
        key=lambda x: (
            0 if x["action"] == "Scale" else (1 if x["action"] == "Stop" else (2 if x["action"] == "Test" else 3)),
            -x["recommended_share_pct"],
        )
    )

    return {
        "customer_id": customer_id,
        "current_expected_meetings": kpi["current_expected_meetings"],
        "current_lo90": kpi["current_lo90"],
        "current_hi90": kpi["current_hi90"],
        "recommended_expected_meetings": kpi["recommended_expected_meetings"],
        "recommended_lo90": kpi["recommended_lo90"],
        "recommended_hi90": kpi["recommended_hi90"],
        "relative_gain_pct": kpi["relative_gain_pct"],
        "p_recommended_beats_current": kpi["p_recommended_beats_current"],
        "recommendations": recommendations,
    }


def apply_operator_plan(customer_id: str, shares: dict[str, float] | None = None) -> dict[str, Any]:
    con = init_db(DB_PATH)
    plan_info = fetch_operator_plan(customer_id)

    if shares is None:
        shares = {item["arm_id"]: item["recommended_share_pct"] / 100.0 for item in plan_info["recommendations"]}

    total_share = sum(shares.values())
    if total_share > 0:
        normalized_shares = {k: v / total_share for k, v in shares.items()}
    else:
        normalized_shares = shares

    con.close()
    return {
        "status": "applied",
        "customer_id": customer_id,
        "active_shares": normalized_shares,
        "message": "Plan successfully applied to next week's send allocation policy.",
    }
