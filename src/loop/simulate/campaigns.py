from typing import Any

import numpy as np
from scipy.special import expit, softmax

from loop.config import DB_PATH, EVAL_GOLD_DIR, LOOP_SEED, ensure_directories
from loop.schema import (
    ALL_ARMS,
    OPENER_TYPES,
    SIGNAL_TYPES,
    CtaType,
    LengthBucket,
    OpenerType,
    ReplyClass,
    SignalType,
    init_db,
)
from loop.simulate.replies import (
    SPLIT_CORE_IDS,
    build_test_dataset,
    build_verification_sample,
    compose_reply_text,
)
from loop.simulate.world import (
    NON_POSITIVE_CLASS_MIX_BY_INDUSTRY,
    CustomerProfile,
    World,
    compute_single_send_positive_prob,
    generate_world,
    sample_signal_age,
)


def compute_historical_arm_propensities(
    profile: CustomerProfile,
) -> dict[str, float]:
    active_signals = [s for s in SIGNAL_TYPES if s != "no_signal"]
    logits = np.array([profile.signal_preference_logits[s] for s in active_signals], dtype=float)
    sig_probs_sub = softmax(logits)

    p_sig: dict[SignalType, float] = {"no_signal": 0.15}
    for idx, s in enumerate(active_signals):
        p_sig[s] = float(0.85 * sig_probs_sub[idx])

    p_op: dict[OpenerType, float] = {}
    for op in OPENER_TYPES:
        if op == profile.dominant_opener:
            p_op[op] = 0.70
        else:
            p_op[op] = 0.15

    propensities: dict[str, float] = {}
    total = 0.0
    for arm in ALL_ARMS:
        s_str, o_str = arm.split(":")
        sig_val: SignalType = s_str
        op_val: OpenerType = o_str
        val = p_sig[sig_val] * p_op[op_val]
        propensities[arm] = val
        total += val

    for arm in ALL_ARMS:
        propensities[arm] = propensities[arm] / total
    return propensities

def simulate_campaigns(world: World, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)

    customers_rows: list[dict[str, Any]] = []
    sends_rows: list[dict[str, Any]] = []
    replies_rows: list[dict[str, Any]] = []

    train_cores = SPLIT_CORE_IDS["train"]
    send_counter = 0
    reply_counter = 0

    base_timestamp = np.datetime64("2026-07-06T09:00:00")

    for cid, profile in world.customers.items():
        customers_rows.append(
            {
                "customer_id": cid,
                "name": profile.name,
                "industry": profile.industry,
                "primary_language": profile.primary_language,
                "n_sends": profile.n_sends,
            }
        )

        propensities = compute_historical_arm_propensities(profile)
        arms_list = list(ALL_ARMS)
        arm_probs = np.array([propensities[a] for a in arms_list], dtype=float)
        arm_probs /= arm_probs.sum()

        non_pos_mix = NON_POSITIVE_CLASS_MIX_BY_INDUSTRY[profile.industry]
        non_pos_classes = list(non_pos_mix.keys())
        non_pos_weights = np.array([non_pos_mix[c] for c in non_pos_classes], dtype=float)
        non_pos_weights /= non_pos_weights.sum()

        sends_per_week = int(np.ceil(profile.n_sends / 12))

        for send_idx in range(profile.n_sends):
            send_counter += 1
            sid = f"send_{send_counter:07d}"
            week = min(12, int(send_idx // sends_per_week) + 1)

            chosen_arm_idx = int(rng.choice(len(arms_list), p=arm_probs))
            arm_id = arms_list[chosen_arm_idx]
            sig_str, op_str = arm_id.split(":")
            sig: SignalType = sig_str
            op: OpenerType = op_str

            propensity_val = float(propensities[arm_id])

            age_days = sample_signal_age(sig, rng)
            if sig == "no_signal":
                icp_fit = float(rng.beta(2.0, 3.0))
            else:
                icp_fit = float(rng.beta(4.0, 2.0))

            cta: CtaType = profile.default_cta
            length: LengthBucket = profile.default_length

            p_pos = compute_single_send_positive_prob(
                profile, sig, age_days, icp_fit, op, cta, length
            )
            pos_true = bool(rng.random() < p_pos)

            if pos_true:
                replied_any = True
                p_meet = float(expit(profile.meeting_logit))
                meeting_true = bool(rng.random() < p_meet)
            else:
                replied_any = bool(rng.random() < profile.p_non_positive_reply)
                meeting_true = False

            sends_rows.append(
                {
                    "send_id": sid,
                    "customer_id": cid,
                    "week": week,
                    "signal_type": sig,
                    "signal_age_days": round(age_days, 2),
                    "icp_fit": round(icp_fit, 4),
                    "opener_type": op,
                    "cta_type": cta,
                    "length_bucket": length,
                    "language": profile.primary_language,
                    "arm_id": arm_id,
                    "policy": "historical_static",
                    "propensity": propensity_val,
                    "replied_any": replied_any,
                    "positive_true": pos_true,
                    "meeting_true": meeting_true,
                }
            )

            if replied_any:
                reply_counter += 1
                rid = f"rep_{reply_counter:07d}"

                if pos_true:
                    reply_class: ReplyClass = "interested" if rng.random() < 0.60 else "info_request"
                else:
                    reply_class = non_pos_classes[int(rng.choice(len(non_pos_classes), p=non_pos_weights))]

                core_id = int(rng.choice(train_cores))
                text, slots = compose_reply_text(
                    reply_class,
                    profile.primary_language,
                    core_id,
                    rng,
                    apply_perturbations=True,
                )

                hour_offset = int(rng.integers(1, 48))
                send_dt = base_timestamp + np.timedelta64((week - 1) * 7, "D") + np.timedelta64(hour_offset, "h")
                received_str = str(send_dt) + "Z"

                revisit_start = None
                revisit_end = None
                if reply_class == "objection_timing":
                    revisit_start = str(send_dt + np.timedelta64(60, "D"))[:10]
                    revisit_end = str(send_dt + np.timedelta64(120, "D"))[:10]

                replies_rows.append(
                    {
                        "reply_id": rid,
                        "send_id": sid,
                        "customer_id": cid,
                        "received_at": received_str,
                        "text": text,
                        "true_class": reply_class,
                        "true_language": profile.primary_language,
                        "referral_name": slots.get("referral_name"),
                        "referral_email": slots.get("referral_email"),
                        "revisit_window_start": revisit_start,
                        "revisit_window_end": revisit_end,
                        "split": "train",
                    }
                )

    return customers_rows, sends_rows, replies_rows

def run_simulation(seed: int = LOOP_SEED) -> None:
    ensure_directories()
    con = init_db(DB_PATH)

    world = generate_world(seed)
    customers, sends, replies = simulate_campaigns(world, seed)

    con.execute("DELETE FROM customers;")
    con.execute("DELETE FROM sends;")
    con.execute("DELETE FROM replies;")

    import pandas as pd

    df_cust = pd.DataFrame(customers)
    df_sends = pd.DataFrame(sends)
    df_replies = pd.DataFrame(replies)

    con.register("df_cust_view", df_cust)
    con.execute("INSERT INTO customers SELECT * FROM df_cust_view;")
    con.unregister("df_cust_view")

    con.register("df_sends_view", df_sends)
    con.execute("INSERT INTO sends SELECT * FROM df_sends_view;")
    con.unregister("df_sends_view")

    con.register("df_replies_view", df_replies)
    con.execute("INSERT INTO replies SELECT * FROM df_replies_view;")
    con.unregister("df_replies_view")

    con.close()

    rng_test = np.random.default_rng(seed + 999)
    test_ds = build_test_dataset(rng_test)
    build_verification_sample(test_ds, EVAL_GOLD_DIR / "verification_sample.csv", 150)

if __name__ == "__main__":
    run_simulation()
