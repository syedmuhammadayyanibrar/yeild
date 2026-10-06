import numpy as np

from loop.schema import REPLY_CLASSES


def compute_f1_score(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    classes: list[str] | tuple[str, ...] = REPLY_CLASSES,
) -> tuple[float, dict[str, dict[str, float]]]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    per_class: dict[str, dict[str, float]] = {}
    f1_list: list[float] = []

    for cls in classes:
        tp = int(np.sum((y_true_arr == cls) & (y_pred_arr == cls)))
        fp = int(np.sum((y_true_arr != cls) & (y_pred_arr == cls)))
        fn = int(np.sum((y_true_arr == cls) & (y_pred_arr != cls)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[str(cls)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": float(tp + fn),
        }
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    return macro_f1, per_class

def compute_confusion_matrix(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    classes: list[str] | tuple[str, ...] = REPLY_CLASSES,
) -> tuple[list[list[int]], list[list[float]]]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    counts = [[0 for _ in range(n)] for _ in range(n)]
    for yt, yp in zip(y_true_arr, y_pred_arr, strict=False):
        if yt in cls_to_idx and yp in cls_to_idx:
            counts[cls_to_idx[yt]][cls_to_idx[yp]] += 1

    normalized: list[list[float]] = []
    for row in counts:
        row_sum = sum(row)
        if row_sum > 0:
            normalized.append([round(c / row_sum, 4) for c in row])
        else:
            normalized.append([0.0 for _ in row])

    return counts, normalized

def compute_ips(
    rewards: np.ndarray,
    behavior_propensities: np.ndarray,
    target_propensities: np.ndarray,
) -> float:
    weights = target_propensities / behavior_propensities
    return float(np.mean(weights * rewards))

def compute_snips(
    rewards: np.ndarray,
    behavior_propensities: np.ndarray,
    target_propensities: np.ndarray,
) -> float:
    weights = target_propensities / behavior_propensities
    denom = np.sum(weights)
    if denom == 0.0:
        return 0.0
    return float(np.sum(weights * rewards) / denom)

def compute_dr(
    rewards: np.ndarray,
    behavior_propensities: np.ndarray,
    target_propensities: np.ndarray,
    q_model_rewards: np.ndarray,
) -> float:
    weights = target_propensities / behavior_propensities
    return float(np.mean(q_model_rewards + weights * (rewards - q_model_rewards)))

def compute_effective_sample_size(
    behavior_propensities: np.ndarray,
    target_propensities: np.ndarray,
) -> float:
    weights = target_propensities / behavior_propensities
    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights**2)
    if sum_w2 == 0.0:
        return 0.0
    return float((sum_w**2) / sum_w2)
