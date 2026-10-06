import csv
import io
from typing import Any

import duckdb
import numpy as np

from loop.attribution.data import SIGNAL_TO_IDX
from loop.attribution.posterior_store import load_posterior
from loop.config import DB_PATH
from loop.ops.scoring import (
    CustomerContext,
    LeadArrays,
    explain_lead,
    meeting_prob_draws,
    signal_half_life_days,
)
from loop.schema import (
    OPENER_TYPES,
    SIGNAL_TYPES,
    SignalType,
    init_db,
)
from loop.simulate.world import SIGNAL_CONFIGS, sample_signal_age

COMPANY_PREFIXES = [
    "Nexus", "Apex", "Synthetix", "Vanguard", "Kinetix", "Aura", "Quantum", "Hyperion",
    "Omni", "Crest", "Vertex", "Solstice", "Prism", "Echo", "Stratum", "Pulse"
]
COMPANY_SUFFIXES = [
    "AI", "Labs", "Systems", "Security", "Analytics", "Networks", "Data", "Cloud",
    "Robotics", "Fintech", "Software", "Tech"
]
TITLES = [
    "VP Engineering", "Chief Technology Officer", "Head of Infrastructure",
    "Director of Data", "VP Product", "Chief Information Security Officer",
    "Head of Growth", "Director of Platform"
]


def ensure_lead_pool_for_customer(customer_id: str, con: duckdb.DuckDBPyConnection) -> None:
    count = con.execute("SELECT COUNT(*) FROM lead_pool WHERE customer_id = ? AND is_sent = false;", [customer_id]).fetchone()[0]
    if count >= 30:
        return

    c_row = con.execute("SELECT primary_language, industry FROM customers WHERE customer_id = ?;", [customer_id]).fetchone()
    if not c_row:
        return
    prim_lang = str(c_row[0])

    rng = np.random.default_rng(hash(customer_id) % (2**31 - 1))
    new_leads = []

    for i in range(50):
        lead_id = f"lead_{customer_id}_{i+1:04d}"
        c_name = f"{rng.choice(COMPANY_PREFIXES)} {rng.choice(COMPANY_SUFFIXES)}"
        title = str(rng.choice(TITLES))
        sig_choice: SignalType = str(rng.choice(SIGNAL_TYPES))
        age = sample_signal_age(sig_choice, rng)
        icp = float(rng.beta(4.0, 2.0)) if sig_choice != "no_signal" else float(rng.beta(2.0, 3.0))

        new_leads.append((
            lead_id, customer_id, c_name, title, sig_choice, round(age, 1), round(icp, 3), prim_lang, False, None, None
        ))

    con.executemany(
        """
        INSERT OR REPLACE INTO lead_pool (
            lead_id, customer_id, company_name, title, signal_type, signal_age_days,
            icp_fit, language, is_sent, allocated_arm, send_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        new_leads,
    )


def fetch_today_leads(customer_id: str, limit: int = 40) -> dict[str, Any]:
    con = init_db(DB_PATH)
    ensure_lead_pool_for_customer(customer_id, con)

    df_leads = con.execute(
        "SELECT * FROM lead_pool WHERE customer_id = ? AND is_sent = false LIMIT 100;",
        [customer_id]
    ).fetchdf()

    if len(df_leads) == 0:
        con.close()
        return {"customer_id": customer_id, "total_available": 0, "leads": []}

    post = load_posterior()
    con.close()

    sig_list = df_leads["signal_type"].tolist()
    age_list = df_leads["signal_age_days"].tolist()
    icp_list = df_leads["icp_fit"].tolist()

    leads_arr = LeadArrays(
        signal_idx=np.array([SIGNAL_TO_IDX[s] for s in sig_list], dtype=int),
        age_days=np.array(age_list, dtype=float),
        icp_fit=np.array(icp_list, dtype=float),
    )

    c_idx = 0
    k_idx = 0
    if post is not None and customer_id in post.customer_ids:
        c_idx = post.customer_index(customer_id)
        k_idx = post.industry_index(customer_id)
    ctx = CustomerContext(customer_idx=c_idx, industry_idx=k_idx)

    if post is not None:
        p_meets_draws = meeting_prob_draws(post, ctx, leads_arr)
        weights = post.weights
        w_p_meets = np.tensordot(weights, p_meets_draws, axes=(0, 0)) * 100.0
    else:
        w_p_meets = np.zeros((len(df_leads), len(OPENER_TYPES)))
        for i in range(len(df_leads)):
            base_r = 3.5
            s_mult = 1.8 if sig_list[i] != "no_signal" else 0.8
            w_p_meets[i, :] = base_r * s_mult

    lead_items = []
    for i in range(len(df_leads)):
        row = df_leads.iloc[i]
        sig_name = str(row["signal_type"])
        age_val = float(row["signal_age_days"])
        icp_val = float(row["icp_fit"])

        best_op_idx = int(np.argmax(w_p_meets[i, :]))
        best_op = OPENER_TYPES[best_op_idx]
        meeting_odds = float(w_p_meets[i, best_op_idx])

        if post is not None:
            terms = explain_lead(post, ctx, leads_arr, i, best_op_idx)
            hl = signal_half_life_days(post, SIGNAL_TO_IDX[sig_name]) if sig_name != "no_signal" else 999.0
        else:
            terms = {"base": -3.2, "icp": (icp_val - 0.5) * 1.2, "signal": 0.8, "opener": 0.3}
            hl = SIGNAL_CONFIGS[sig_name][1] * np.log(2.0) if sig_name != "no_signal" else 999.0

        stale_friday = (age_val + 4.0) > hl and sig_name != "no_signal"

        if sig_name == "no_signal":
            why = f"Cold outreach baseline ({meeting_odds:.1f}/100); low ICP leverage ({icp_val:.2f})."
        elif stale_friday:
            why = f"Fresh {sig_name.replace('_', ' ')} (+{terms.get('signal', 0.5):.1f} lift); goes stale in ~{max(1, int(hl - age_val))}d."
        else:
            why = f"High fit ({icp_val:.2f}) with {sig_name.replace('_', ' ')}; durable signal ({hl:.0f}d half-life)."

        lead_items.append({
            "lead_id": str(row["lead_id"]),
            "company_name": str(row["company_name"]),
            "title": str(row["title"]),
            "signal_type": sig_name,
            "signal_age_days": age_val,
            "icp_fit": icp_val,
            "opener_type": best_op,
            "expected_meetings_per_100": round(meeting_odds, 2),
            "log_odds_terms": {k: round(v, 2) for k, v in terms.items()},
            "stale_by_friday": stale_friday,
            "why": why,
        })

    lead_items.sort(key=lambda x: x["expected_meetings_per_100"], reverse=True)
    return {
        "customer_id": customer_id,
        "total_available": len(lead_items),
        "leads": lead_items[:limit],
    }


def generate_leads_csv(customer_id: str) -> str:
    data = fetch_today_leads(customer_id, limit=100)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "LeadID", "Company", "Title", "SignalType", "SignalAgeDays",
        "ICPFit", "RecommendedOpener", "ExpectedMeetingsPer100", "StaleByFriday", "Why"
    ])
    for item in data["leads"]:
        writer.writerow([
            item["lead_id"],
            item["company_name"],
            item["title"],
            item["signal_type"],
            item["signal_age_days"],
            item["icp_fit"],
            item["opener_type"],
            item["expected_meetings_per_100"],
            "Yes" if item["stale_by_friday"] else "No",
            item["why"],
        ])
    return output.getvalue()
