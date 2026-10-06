import csv
import json
from typing import Any

import numpy as np

from loop.config import (
    ARTIFACTS_DIR,
    EVAL_GOLD_DIR,
)
from loop.evaluation.bootstrap import (
    bootstrap_ci,
)
from loop.evaluation.metrics import (
    compute_confusion_matrix,
    compute_f1_score,
)
from loop.reply_intel.baseline import BaselineClassifier, train_baseline
from loop.reply_intel.calibration import (
    apply_temperature,
    compute_brier,
    compute_ece,
    compute_nll,
    fit_temperature,
)
from loop.reply_intel.drafts import generate_draft_response
from loop.reply_intel.llm_classifier import classify_with_llm
from loop.reply_intel.llm_client import LlmClient
from loop.reply_intel.router import (
    build_calibration_dataset,
    choose_routing_threshold,
    compute_risk_coverage_curve,
)
from loop.reply_intel.taxonomy import CLASS_TO_INDEX, INDEX_TO_CLASS
from loop.reply_intel.validators import run_all_validators
from loop.rng import get_rng
from loop.schema import REPLY_CLASSES
from loop.simulate.replies import build_hard_dataset, build_test_dataset


def evaluate_reply_intelligence(
    seed: int = 42,
    fast_mode: bool = False,
) -> dict[str, Any]:
    rng = get_rng(seed)

    model_path = ARTIFACTS_DIR / "baseline_model.joblib"
    if model_path.exists():
        clf = BaselineClassifier.load(model_path)
    else:
        clf = train_baseline()
        clf.save(model_path)

    test_items = build_test_dataset(rng)
    hard_items = build_hard_dataset(rng)

    calib_items = build_calibration_dataset(seed=seed, n_per_cell=15)
    calib_texts = [str(x["text"]) for x in calib_items]
    calib_true_labels = [str(x["true_class"]) for x in calib_items]
    calib_true_indices = np.array([CLASS_TO_INDEX[c] for c in calib_true_labels])

    raw_calib_probs = clf.predict_proba(calib_texts)
    temp = fit_temperature(raw_calib_probs, calib_true_indices)
    calib_probs = apply_temperature(raw_calib_probs, temp)
    tau = choose_routing_threshold(calib_probs, calib_true_indices, max_error_rate=0.05)

    test_texts = [str(x["text"]) for x in test_items]
    test_true_labels = [str(x["true_class"]) for x in test_items]
    test_true_indices = np.array([CLASS_TO_INDEX[c] for c in test_true_labels])

    raw_test_probs = clf.predict_proba(test_texts)
    calib_test_probs = apply_temperature(raw_test_probs, temp)
    baseline_test_preds = [INDEX_TO_CLASS[i] for i in np.argmax(calib_test_probs, axis=1)]

    macro_f1, per_class = compute_f1_score(test_true_labels, baseline_test_preds)

    def f1_stat(idx_arr: np.ndarray) -> float:
        sub_true = [test_true_labels[i] for i in idx_arr]
        sub_pred = [baseline_test_preds[i] for i in idx_arr]
        f1, _ = compute_f1_score(sub_true, sub_pred)
        return f1

    _, f1_lo, f1_hi = bootstrap_ci(np.arange(len(test_items)), f1_stat, n_resamples=1000, seed=seed)

    counts_mat, norm_mat = compute_confusion_matrix(test_true_labels, baseline_test_preds)

    hard_texts = [str(x["text"]) for x in hard_items]
    hard_true_labels = [str(x["true_class"]) for x in hard_items]
    raw_hard_probs = clf.predict_proba(hard_texts)
    calib_hard_probs = apply_temperature(raw_hard_probs, temp)
    baseline_hard_preds = [INDEX_TO_CLASS[i] for i in np.argmax(calib_hard_probs, axis=1)]
    hard_macro_f1, _ = compute_f1_score(hard_true_labels, baseline_hard_preds)

    handwritten_file = EVAL_GOLD_DIR / "handwritten.jsonl"
    handwritten_macro_f1 = None
    if handwritten_file.exists():
        hw_items = []
        with open(handwritten_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    hw_items.append(json.loads(line))
        if hw_items:
            hw_texts = [x["text"] for x in hw_items]
            hw_labels = [x["label"] for x in hw_items]
            hw_preds = clf.predict(hw_texts)
            hw_f1, _ = compute_f1_score(hw_labels, hw_preds)
            handwritten_macro_f1 = hw_f1

    verification_file = EVAL_GOLD_DIR / "verification_sample.csv"
    verification_agreement = None
    if verification_file.exists():
        with open(verification_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            v_rows = [row for row in reader if row.get("human_label", "").strip()]
            if v_rows:
                agree = sum(1 for r in v_rows if r["human_label"] == r["construction_label"])
                verification_agreement = float(agree / len(v_rows))

    ece_before, _ = compute_ece(raw_test_probs, test_true_indices, n_bins=15)
    ece_after, rel_bins = compute_ece(calib_test_probs, test_true_indices, n_bins=15)
    nll_before = compute_nll(raw_test_probs, test_true_indices)
    nll_after = compute_nll(calib_test_probs, test_true_indices)
    brier_before = compute_brier(raw_test_probs, test_true_indices)
    brier_after = compute_brier(calib_test_probs, test_true_indices)

    aurc, rc_points = compute_risk_coverage_curve(calib_test_probs, test_true_indices)
    test_conf = np.max(calib_test_probs, axis=1)
    auto_mask = test_conf >= tau
    actual_coverage = float(np.mean(auto_mask))
    test_true_arr = np.array(test_true_labels)
    baseline_pred_arr = np.array(baseline_test_preds)
    actual_risk = float(np.mean(baseline_pred_arr[auto_mask] != test_true_arr[auto_mask])) if actual_coverage > 0 else 0.0

    routing_point = {
        "threshold": tau,
        "coverage": actual_coverage,
        "risk": actual_risk,
    }

    validator_passes = []
    for item in test_items:
        draft = generate_draft_response(
            reply_text=item["text"],
            reply_class=item["true_class"],
            language=item["true_language"],
            referral_name=item["referral_name"],
            referral_email=item["referral_email"],
        )
        vres = run_all_validators(
            draft=draft,
            reply_class=item["true_class"],
            language=item["true_language"],
            context_text=item["text"],
            referral_name=item["referral_name"],
            referral_email=item["referral_email"],
        )
        validator_passes.append(all(vres.values()))
    draft_pass_rate = float(np.mean(validator_passes))

    injection_items = [it for it in hard_items if it.get("is_injection")]
    attack_success_count = 0
    clean_correct_count = 0
    for it in injection_items:
        pred = clf.predict([it["text"]])[0]
        if pred == it["target_class"] and pred != it["true_class"]:
            attack_success_count += 1
        clean_text = it["text"].split("\n\nSystem override:")[0].split("\n\n[ADMIN INSTRUCTION]:")[0]
        clean_pred = clf.predict([clean_text])[0]
        if clean_pred == it["true_class"]:
            clean_correct_count += 1

    attack_success_rate = float(attack_success_count / len(injection_items)) if injection_items else 0.0
    clean_acc = float(clean_correct_count / len(injection_items)) if injection_items else 0.0

    llm_client = LlmClient()
    llm_available = llm_client.is_available()

    llm_macro_f1 = None
    llm_hard_f1 = None
    llm_attack_success = None
    paired_diff = None

    if llm_available and not fast_mode:
        llm_preds = []
        for it in test_items[:60]:
            res = classify_with_llm(it["text"], client=llm_client)
            if res:
                llm_preds.append(res["predicted_class"])
            else:
                llm_preds.append(clf.predict([it["text"]])[0])
        llm_macro_f1, _ = compute_f1_score([t["true_class"] for t in test_items[:60]], llm_preds)

    return {
        "baseline": {
            "test_macro_f1": macro_f1,
            "test_macro_f1_ci": [f1_lo, f1_hi],
            "hard_macro_f1": hard_macro_f1,
            "handwritten_macro_f1": handwritten_macro_f1,
            "verification_agreement": verification_agreement,
            "per_class": per_class,
            "temperature": temp,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "nll_before": nll_before,
            "nll_after": nll_after,
            "brier_before": brier_before,
            "brier_after": brier_after,
            "tau": tau,
            "aurc": aurc,
            "actual_coverage": actual_coverage,
            "actual_risk": actual_risk,
            "draft_pass_rate": draft_pass_rate,
            "attack_success_rate": attack_success_rate,
            "clean_accuracy": clean_acc,
        },
        "llm": {
            "available": llm_available,
            "test_macro_f1": llm_macro_f1,
            "hard_macro_f1": llm_hard_f1,
            "attack_success_rate": llm_attack_success,
            "paired_diff_vs_baseline": paired_diff,
        },
        "plots": {
            "confusion_matrix": {
                "counts": counts_mat,
                "normalized": norm_mat,
                "labels": list(REPLY_CLASSES),
            },
            "reliability_bins": rel_bins,
            "risk_coverage_points": rc_points,
            "routing_point": routing_point,
        },
    }
