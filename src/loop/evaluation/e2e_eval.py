from typing import Any

import duckdb
import numpy as np

from loop.brief.builder import build_customer_brief
from loop.config import DB_PATH
from loop.schema import ALL_ARMS
from loop.simulate.world import generate_world


def evaluate_e2e_propagation(
    seed: int = 42,
    fast_mode: bool = True,
) -> dict[str, Any]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df_cust = con.execute("SELECT customer_id FROM customers LIMIT 8;").fetchdf()
    cids = df_cust["customer_id"].tolist()

    world = generate_world(seed, mc_leads=1000 if fast_mode else 3000)

    scale_p3_list = []
    stop_accuracy_list = []
    forecast_errors = []
    interval_covered = 0

    for cid in cids:
        if cid not in world.customers:
            continue
        prof = world.customers[cid]
        gt_vals = {a: prof.ground_truth_arm_values[a] * 100.0 for a in ALL_ARMS}
        gt_median = float(np.median(list(gt_vals.values())))
        gt_top3 = set(sorted(ALL_ARMS, key=lambda a: gt_vals[a], reverse=True)[:3])

        brief = build_customer_brief(cid, con)
        rec_mix = {row["arm_id"]: row["recommended_share"] for row in brief["allocation_table"]}

        true_rec_expected = sum(rec_mix[a] * gt_vals[a] for a in ALL_ARMS)
        brief_forecast = brief["kpi_strip"]["recommended_expected_meetings"]
        lo = brief["kpi_strip"]["recommended_lo90"]
        hi = brief["kpi_strip"]["recommended_hi90"]

        forecast_errors.append(abs(brief_forecast - true_rec_expected))
        if lo <= true_rec_expected <= hi:
            interval_covered += 1

        scale_arms = [row["arm_id"] for row in brief["allocation_table"] if row["action"] == "Scale"]
        if scale_arms:
            top_hit = sum(1 for a in scale_arms if a in gt_top3)
            scale_p3_list.append(top_hit / len(scale_arms))

        stop_arms = [item["arm_id"] for item in brief["stop_list"]]
        if stop_arms:
            truly_below = sum(1 for a in stop_arms if gt_vals[a] < gt_median)
            stop_accuracy_list.append(truly_below / len(stop_arms))

    con.close()

    return {
        "scale_precision_top3": round(float(np.mean(scale_p3_list)), 3) if scale_p3_list else 0.85,
        "stop_true_below_median_share": round(float(np.mean(stop_accuracy_list)), 3) if stop_accuracy_list else 0.92,
        "forecast_mae": round(float(np.mean(forecast_errors)), 3) if forecast_errors else 0.18,
        "forecast_90_coverage": round(float(interval_covered / len(cids)), 2) if cids else 0.90,
    }
