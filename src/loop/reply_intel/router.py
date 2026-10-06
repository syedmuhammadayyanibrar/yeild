import json
from typing import Any

import numpy as np

from loop.config import ARTIFACTS_DIR, DB_PATH
from loop.reply_intel.baseline import BaselineClassifier, train_baseline
from loop.reply_intel.calibration import apply_temperature, fit_temperature
from loop.reply_intel.taxonomy import CLASS_TO_INDEX, INDEX_TO_CLASS
from loop.rng import get_rng
from loop.schema import REPLY_CLASSES, init_db
from loop.simulate.replies import SPLIT_CORE_IDS, compose_reply_text


def compute_risk_coverage_curve(
    probs: np.ndarray,
    true_indices: np.ndarray,
    n_points: int = 50,
) -> tuple[float, list[dict[str, float]]]:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    errors = (predictions != true_indices).astype(float)

    thresholds = np.linspace(0.0, 1.0, n_points)
    curve_points: list[dict[str, float]] = []
    n_total = len(confidences)

    coverages: list[float] = []
    risks: list[float] = []

    for th in thresholds:
        auto_mask = confidences >= th
        n_auto = int(np.sum(auto_mask))
        if n_auto == 0:
            coverage = 0.0
            risk = 0.0
        else:
            coverage = float(n_auto / n_total)
            risk = float(np.mean(errors[auto_mask]))
        coverages.append(coverage)
        risks.append(risk)
        curve_points.append(
            {
                "threshold": float(th),
                "coverage": coverage,
                "risk": risk,
            }
        )

    sort_order = np.argsort(coverages)
    sorted_cov = np.array(coverages)[sort_order]
    sorted_risk = np.array(risks)[sort_order]
    aurc = float(np.trapezoid(sorted_risk, sorted_cov))

    return aurc, curve_points

def choose_routing_threshold(
    probs: np.ndarray,
    true_indices: np.ndarray,
    max_error_rate: float = 0.05,
) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    errors = (predictions != true_indices).astype(float)

    candidates = np.sort(np.unique(confidences))
    selected_threshold = 1.0

    for th in candidates:
        auto_mask = confidences >= th
        n_auto = np.sum(auto_mask)
        if n_auto == 0:
            continue
        err_rate = float(np.mean(errors[auto_mask]))
        if err_rate <= max_error_rate:
            selected_threshold = float(th)
            break

    return selected_threshold

def build_calibration_dataset(seed: int = 42, n_per_cell: int = 15) -> list[dict[str, Any]]:
    rng = get_rng(seed)
    calib_cores = SPLIT_CORE_IDS["calibration"]
    items: list[dict[str, Any]] = []
    from loop.schema import LANGUAGES

    for cls in REPLY_CLASSES:
        for lang in LANGUAGES:
            for i in range(n_per_cell):
                core_id = calib_cores[i % len(calib_cores)]
                text, slots = compose_reply_text(cls, lang, core_id, rng, apply_perturbations=True)
                items.append(
                    {
                        "text": text,
                        "true_class": cls,
                        "true_language": lang,
                        "referral_name": slots.get("referral_name"),
                        "referral_email": slots.get("referral_email"),
                    }
                )
    return items

def run_classification_pipeline() -> None:
    import pandas as pd

    from loop.reply_intel.drafts import generate_draft_response
    from loop.reply_intel.validators import run_all_validators

    model_path = ARTIFACTS_DIR / "baseline_model.joblib"
    if model_path.exists():
        clf = BaselineClassifier.load(model_path)
    else:
        clf = train_baseline()
        clf.save(model_path)

    calib_items = build_calibration_dataset(seed=42)
    calib_texts = [str(x["text"]) for x in calib_items]
    calib_labels = [str(x["true_class"]) for x in calib_items]
    calib_true_idx = np.array([CLASS_TO_INDEX[c] for c in calib_labels])

    raw_calib_probs = clf.predict_proba(calib_texts)
    temp = fit_temperature(raw_calib_probs, calib_true_idx)
    calib_probs = apply_temperature(raw_calib_probs, temp)
    tau = choose_routing_threshold(calib_probs, calib_true_idx, max_error_rate=0.05)

    meta = {
        "temperature": temp,
        "tau": tau,
    }
    with open(ARTIFACTS_DIR / "classifier_calibration.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    con = init_db(DB_PATH)
    df_replies = con.execute("SELECT * FROM replies;").fetchdf()

    if len(df_replies) == 0:
        con.close()
        return

    texts = df_replies["text"].tolist()
    raw_probs = clf.predict_proba(texts)
    calibrated_probs = apply_temperature(raw_probs, temp)
    confidences = np.max(calibrated_probs, axis=1)
    pred_indices = np.argmax(calibrated_probs, axis=1)

    rows: list[dict[str, Any]] = []
    for i, row in df_replies.iterrows():
        p_row = calibrated_probs[i]
        cls_probs_map = {REPLY_CLASSES[j]: float(p_row[j]) for j in range(len(REPLY_CLASSES))}
        pred_cls = INDEX_TO_CLASS[int(pred_indices[i])]
        conf = float(confidences[i])
        routed = bool(conf < tau)

        draft = generate_draft_response(
            reply_text=str(row["text"]),
            reply_class=pred_cls,
            language=str(row["true_language"]),
            referral_name=row["referral_name"],
            referral_email=row["referral_email"],
        )

        validators_res = run_all_validators(
            draft=draft,
            reply_class=pred_cls,
            language=str(row["true_language"]),
            context_text=str(row["text"]),
            referral_name=row["referral_name"],
            referral_email=row["referral_email"],
        )

        rows.append(
            {
                "reply_id": row["reply_id"],
                "model": "baseline_calibrated",
                "class_probs_json": json.dumps(cls_probs_map),
                "predicted_class": pred_cls,
                "confidence": conf,
                "routed_to_review": routed,
                "language_detected": str(row["true_language"]),
                "referral_name": row["referral_name"],
                "referral_email": row["referral_email"],
                "revisit_date": row["revisit_window_start"],
                "draft_reply": draft,
                "validators_json": json.dumps(validators_res),
            }
        )

    con.execute("DELETE FROM classifications;")
    df_class = pd.DataFrame(rows)
    con.register("df_class_view", df_class)
    con.execute("INSERT INTO classifications SELECT * FROM df_class_view;")
    con.unregister("df_class_view")
    con.close()

if __name__ == "__main__":
    run_classification_pipeline()
