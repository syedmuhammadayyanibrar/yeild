import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax


def compute_nll(probs: np.ndarray, true_indices: np.ndarray) -> float:
    eps = 1e-12
    clipped = np.clip(probs, eps, 1.0)
    n_samples = len(true_indices)
    log_probs = np.log(clipped[np.arange(n_samples), true_indices])
    return float(-np.mean(log_probs))

def compute_brier(probs: np.ndarray, true_indices: np.ndarray) -> float:
    n_samples, n_classes = probs.shape
    one_hot = np.zeros((n_samples, n_classes), dtype=float)
    one_hot[np.arange(n_samples), true_indices] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

def compute_ece(
    probs: np.ndarray,
    true_indices: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, list[dict[str, float]]]:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == true_indices).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_stats: list[dict[str, float]] = []
    n_samples = len(confidences)

    for lower, upper in zip(bin_lowers, bin_uppers, strict=True):
        in_bin = (confidences > lower) & (confidences <= upper)
        bin_size = int(np.sum(in_bin))
        if bin_size > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)
            bin_stats.append(
                {
                    "bin_lower": float(lower),
                    "bin_upper": float(upper),
                    "confidence": bin_conf,
                    "accuracy": bin_acc,
                    "count": float(bin_size),
                }
            )
        else:
            bin_stats.append(
                {
                    "bin_lower": float(lower),
                    "bin_upper": float(upper),
                    "confidence": float((lower + upper) / 2),
                    "accuracy": 0.0,
                    "count": 0.0,
                }
            )

    return float(ece), bin_stats

def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    eps = 1e-12
    safe_temp = max(0.05, float(temperature))
    log_p = np.log(np.clip(probs, eps, 1.0))
    scaled = log_p / safe_temp
    return softmax(scaled, axis=1)

def fit_temperature(probs: np.ndarray, true_indices: np.ndarray) -> float:
    eps = 1e-12
    log_p = np.log(np.clip(probs, eps, 1.0))

    def objective(t: np.ndarray) -> float:
        temp = float(t[0])
        scaled = log_p / temp
        p_scaled = softmax(scaled, axis=1)
        return compute_nll(p_scaled, true_indices)

    res = minimize(
        objective,
        x0=np.array([1.0]),
        bounds=[(0.05, 10.0)],
        method="L-BFGS-B",
    )
    return float(res.x[0])
