import json
from pathlib import Path
from typing import Any

import duckdb
import jax.numpy as jnp
import numpy as np

from loop.schema import (
    CTA_TYPES,
    INDUSTRIES,
    LENGTH_BUCKETS,
    OPENER_TYPES,
    POSITIVE_CLASSES,
    SIGNAL_TYPES,
)

INDUSTRY_TO_IDX: dict[str, int] = {ind: i for i, ind in enumerate(INDUSTRIES)}
SIGNAL_TO_IDX: dict[str, int] = {sig: i for i, sig in enumerate(SIGNAL_TYPES)}
OPENER_TO_IDX: dict[str, int] = {op: i for i, op in enumerate(OPENER_TYPES)}
CTA_TO_IDX: dict[str, int] = {cta: i for i, cta in enumerate(CTA_TYPES)}
LENGTH_TO_IDX: dict[str, int] = {lb: i for i, lb in enumerate(LENGTH_BUCKETS)}

def prepare_attribution_data(
    db_path: Path,
    label_type: str = "soft",
    max_customers: int | None = None,
    sends_per_customer_limit: int | None = None,
) -> dict[str, Any]:
    con = duckdb.connect(str(db_path))

    df_cust = con.execute("SELECT customer_id, industry FROM customers ORDER BY customer_id;").fetchdf()
    if max_customers is not None:
        df_cust = df_cust.head(max_customers)

    valid_cids = set(df_cust["customer_id"].tolist())
    cust_id_list = sorted(list(valid_cids))
    cust_to_idx = {cid: i for i, cid in enumerate(cust_id_list)}
    cust_industry_idx = np.array([INDUSTRY_TO_IDX[row["industry"]] for _, row in df_cust.iterrows()])

    sends_query = """
    SELECT 
        s.send_id,
        s.customer_id,
        s.signal_type,
        s.signal_age_days,
        s.icp_fit,
        s.opener_type,
        s.cta_type,
        s.length_bucket,
        s.replied_any,
        s.positive_true,
        s.meeting_true,
        c.predicted_class,
        c.class_probs_json
    FROM sends s
    LEFT JOIN replies r ON s.send_id = r.send_id
    LEFT JOIN classifications c ON r.reply_id = c.reply_id
    ORDER BY s.customer_id, s.week, s.send_id;
    """
    df_sends = con.execute(sends_query).fetchdf()
    con.close()

    df_sends = df_sends[df_sends["customer_id"].isin(valid_cids)]

    if sends_per_customer_limit is not None:
        df_sends = df_sends.groupby("customer_id").head(sends_per_customer_limit).reset_index(drop=True)

    n_sends = len(df_sends)

    cust_indices = np.array([cust_to_idx[cid] for cid in df_sends["customer_id"]], dtype=int)
    sig_indices = np.array([SIGNAL_TO_IDX[st] for st in df_sends["signal_type"]], dtype=int)
    op_indices = np.array([OPENER_TO_IDX[op] for op in df_sends["opener_type"]], dtype=int)
    cta_indices = np.array([CTA_TO_IDX[cta] for cta in df_sends["cta_type"]], dtype=int)
    len_indices = np.array([LENGTH_TO_IDX[lb] for lb in df_sends["length_bucket"]], dtype=int)

    ages = df_sends["signal_age_days"].to_numpy(dtype=float)
    icp = df_sends["icp_fit"].to_numpy(dtype=float)
    meeting_true = df_sends["meeting_true"].to_numpy(dtype=float)

    labels = np.zeros(n_sends, dtype=float)
    for idx, row in df_sends.iterrows():
        if not row["replied_any"]:
            labels[idx] = 0.0
            continue

        if label_type == "oracle":
            labels[idx] = 1.0 if row["positive_true"] else 0.0
        elif label_type == "hard":
            pred = row["predicted_class"]
            labels[idx] = 1.0 if pred in POSITIVE_CLASSES else 0.0
        else:
            probs_str = row["class_probs_json"]
            if probs_str and isinstance(probs_str, str):
                try:
                    probs_map = json.loads(probs_str)
                    soft_val = sum(float(probs_map.get(c, 0.0)) for c in POSITIVE_CLASSES)
                    labels[idx] = float(soft_val)
                except Exception:
                    labels[idx] = 1.0 if row["positive_true"] else 0.0
            else:
                labels[idx] = 1.0 if row["positive_true"] else 0.0

    return {
        "n_customers": len(cust_id_list),
        "n_industries": len(INDUSTRIES),
        "n_signals": len(SIGNAL_TYPES),
        "n_openers": len(OPENER_TYPES),
        "n_ctas": len(CTA_TYPES),
        "n_lengths": len(LENGTH_BUCKETS),
        "customer_ids": cust_id_list,
        "cust_to_idx": cust_to_idx,
        "cust_industry": jnp.array(cust_industry_idx),
        "customer_idx": jnp.array(cust_indices),
        "signal_idx": jnp.array(sig_indices),
        "opener_idx": jnp.array(op_indices),
        "cta_idx": jnp.array(cta_indices),
        "length_idx": jnp.array(len_indices),
        "age_days": jnp.array(ages),
        "icp_fit": jnp.array(icp),
        "y": jnp.array(labels),
        "meeting_true": jnp.array(meeting_true),
        "replied_any": jnp.array(df_sends["replied_any"].to_numpy(dtype=float)),
    }
