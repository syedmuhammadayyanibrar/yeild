from datetime import datetime
from typing import Any

import duckdb

from loop.config import DB_PATH
from loop.schema import init_db


def fetch_inbox_data(customer_id: str, con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    query = """
    SELECT 
        r.reply_id,
        r.send_id,
        r.customer_id,
        r.received_at,
        r.text,
        r.true_class,
        r.true_language,
        COALESCE(c.predicted_class, r.true_class) as predicted_class,
        COALESCE(c.confidence, 0.90) as confidence,
        COALESCE(c.routed_to_review, false) as routed_to_review,
        c.referral_name,
        c.referral_email,
        c.revisit_date,
        c.draft_reply,
        a.action as action_taken
    FROM replies r
    LEFT JOIN classifications c ON r.reply_id = c.reply_id
    LEFT JOIN actions a ON r.reply_id = a.reply_id
    WHERE r.customer_id = ?
    ORDER BY r.received_at DESC
    LIMIT 100;
    """
    df = con.execute(query, [customer_id]).fetchdf()

    needs_you = []
    ready_to_send = []
    done_automatically = []

    for _, row in df.iterrows():
        p_cls = str(row["predicted_class"])
        routed = bool(row["routed_to_review"])
        conf = float(row["confidence"])
        action = str(row["action_taken"]) if row["action_taken"] is not None else None
        if action == "approve":
            cat = "done_automatically"
        elif p_cls in {"unsubscribe", "out_of_office", "referral"} and not routed:
            cat = "done_automatically"
        elif p_cls in {"interested", "objection_competitor"} or routed or conf < 0.85:
            cat = "needs_you"
        else:
            cat = "ready_to_send"

        draft_val = str(row["draft_reply"]) if row["draft_reply"] else None
        if draft_val is None:
            if p_cls == "interested":
                draft_val = "Thanks for following up! Would Thursday at 2:00 PM or Friday morning work for a brief 15-minute intro?"
            elif p_cls == "objection_budget":
                draft_val = "Understood on the budget timing. Would you be open to a quick 2-page case study on how we reduced outbound CAC by 35%?"
            elif p_cls == "objection_competitor":
                draft_val = "Makes sense. Many teams using that platform pair us alongside for Bayesian signal routing. Happy to show a 5-minute teardown if useful."
            elif p_cls == "info_request":
                draft_val = "Here is an overview of how Yeild models signal half-life and opener lift. Would a quick walkthrough this week be helpful?"

        item = {
            "reply_id": str(row["reply_id"]),
            "send_id": str(row["send_id"]),
            "customer_id": str(row["customer_id"]),
            "received_at": str(row["received_at"]),
            "text": str(row["text"]),
            "predicted_class": p_cls,
            "confidence": conf,
            "routed_to_review": routed,
            "language": str(row["true_language"]),
            "draft_reply": draft_val,
            "referral_name": str(row["referral_name"]) if row["referral_name"] else None,
            "referral_email": str(row["referral_email"]) if row["referral_email"] else None,
            "revisit_date": str(row["revisit_date"]) if row["revisit_date"] else None,
            "category": cat,
            "action_taken": action,
        }

        if cat == "needs_you":
            needs_you.append(item)
        elif cat == "ready_to_send":
            ready_to_send.append(item)
        else:
            done_automatically.append(item)

    return {
        "total": len(df),
        "needs_you_count": len(needs_you),
        "ready_to_send_count": len(ready_to_send),
        "done_automatically_count": len(done_automatically),
        "needs_you": needs_you,
        "ready_to_send": ready_to_send,
        "done_automatically": done_automatically,
    }


def record_inbox_action(
    reply_id: str,
    customer_id: str,
    action: str,
    final_text: str | None,
    corrected_class: str | None,
) -> dict[str, Any]:
    con = init_db(DB_PATH)
    now_str = datetime.utcnow().isoformat() + "Z"
    text_val = final_text or ""
    class_val = corrected_class or ""

    con.execute(
        """
        INSERT OR REPLACE INTO actions (reply_id, customer_id, action, final_text, corrected_class, created_at)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        [reply_id, customer_id, action, text_val, class_val, now_str],
    )

    if action == "approve" or action == "send":
        r_row = con.execute("SELECT true_class, referral_name, referral_email, revisit_window_start FROM replies WHERE reply_id = ?;", [reply_id]).fetchone()
        if r_row:
            cls_name = r_row[0]
            if cls_name == "unsubscribe":
                con.execute(
                    "INSERT OR REPLACE INTO suppression_list (email_or_domain, customer_id, reason, created_at) VALUES (?, ?, ?, ?);",
                    [f"contact_{reply_id}@suppressed.com", customer_id, "unsubscribe", now_str],
                )
            elif cls_name == "referral" and r_row[2]:
                con.execute(
                    "INSERT OR REPLACE INTO referral_leads (lead_id, customer_id, name, email, source_reply_id, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                    [f"ref_{reply_id}", customer_id, r_row[1], r_row[2], reply_id, now_str],
                )
            elif cls_name == "objection_timing" and r_row[3]:
                con.execute(
                    "INSERT OR REPLACE INTO followups (reply_id, customer_id, contact_hint, scheduled_date, status) VALUES (?, ?, ?, ?, ?);",
                    [reply_id, customer_id, "followup", r_row[3], "scheduled"],
                )

    con.close()
    return {"status": "ok", "reply_id": reply_id, "action": action}
