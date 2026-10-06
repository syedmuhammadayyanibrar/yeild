from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from loop.config import ARTIFACTS_DIR

POSTERIOR_PATH = ARTIFACTS_DIR / "posterior.npz"
POSTERIOR_BASE_PATH = ARTIFACTS_DIR / "posterior_base.npz"
PARAMETER_NAMES: tuple[str, ...] = (
    "alpha_c",
    "beta_cs",
    "tau_s",
    "a2_c",
    "delta",
    "eta",
    "lambda_len",
    "gamma",
)


@dataclass
class Posterior:
    draws: dict[str, np.ndarray]
    weights: np.ndarray
    customer_ids: list[str]
    cust_industry: np.ndarray
    method: str
    updates_applied: int

    @property
    def n_draws(self) -> int:
        return int(self.weights.shape[0])

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights**2))

    def customer_index(self, customer_id: str) -> int:
        return self.customer_ids.index(customer_id)

    def industry_index(self, customer_id: str) -> int:
        return int(self.cust_industry[self.customer_index(customer_id)])

    def sample_indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.n_draws, size=n, p=self.weights)


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]) -> tuple[float, ...]:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative = cumulative / cumulative[-1]
    return tuple(float(sorted_values[min(int(np.searchsorted(cumulative, q)), len(sorted_values) - 1)]) for q in quantiles)


def weighted_interval(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    lo, med, hi = weighted_quantiles(values, weights, (0.05, 0.5, 0.95))
    return {"median": med, "lo90": lo, "hi90": hi}


def weighted_probability(mask: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * mask.astype(float)))


def posterior_from_fit(fit_res: dict[str, Any]) -> Posterior:
    samples = fit_res["samples"]
    data = fit_res["data"]
    draws = {name: np.asarray(samples[name], dtype=np.float64) for name in PARAMETER_NAMES}
    n_draws = draws["alpha_c"].shape[0]
    draws["gamma"] = draws["gamma"].reshape(n_draws)
    return Posterior(
        draws=draws,
        weights=np.full(n_draws, 1.0 / n_draws),
        customer_ids=[str(c) for c in data["customer_ids"]],
        cust_industry=np.asarray(data["cust_industry"], dtype=int),
        method=str(fit_res["method"]),
        updates_applied=0,
    )


def save_posterior(posterior: Posterior, path: Path = POSTERIOR_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weights=posterior.weights,
        customer_ids=np.array(posterior.customer_ids),
        cust_industry=posterior.cust_industry,
        method=np.array(posterior.method),
        updates_applied=np.array(posterior.updates_applied),
        **{f"draw_{name}": posterior.draws[name] for name in PARAMETER_NAMES},
    )


def load_posterior(path: Path = POSTERIOR_PATH) -> Posterior | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        return Posterior(
            draws={name: np.asarray(archive[f"draw_{name}"]) for name in PARAMETER_NAMES},
            weights=np.asarray(archive["weights"]),
            customer_ids=[str(c) for c in archive["customer_ids"]],
            cust_industry=np.asarray(archive["cust_industry"], dtype=int),
            method=str(archive["method"]),
            updates_applied=int(archive["updates_applied"]),
        )


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = weights.shape[0]
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def apply_importance_update(
    posterior: Posterior,
    log_likelihood_per_draw: np.ndarray,
    rng: np.random.Generator,
    resample_below_fraction: float = 0.5,
) -> tuple[Posterior, float, bool]:
    log_w = np.log(np.maximum(posterior.weights, 1e-300)) + log_likelihood_per_draw
    log_w = log_w - np.max(log_w)
    new_weights = np.exp(log_w)
    new_weights = new_weights / new_weights.sum()
    ess = float(1.0 / np.sum(new_weights**2))
    resampled = ess < resample_below_fraction * posterior.n_draws
    if resampled:
        idx = systematic_resample(new_weights, rng)
        draws = {name: arr[idx] for name, arr in posterior.draws.items()}
        new_weights = np.full(posterior.n_draws, 1.0 / posterior.n_draws)
    else:
        draws = posterior.draws
    updated = Posterior(
        draws=draws,
        weights=new_weights,
        customer_ids=posterior.customer_ids,
        cust_industry=posterior.cust_industry,
        method=posterior.method,
        updates_applied=posterior.updates_applied + 1,
    )
    return updated, ess, resampled
