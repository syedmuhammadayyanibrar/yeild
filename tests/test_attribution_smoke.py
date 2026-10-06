import jax.numpy as jnp
import numpy as np

from loop.attribution.inference import fit


def test_attribution_smoke_recovers_sign_and_rank() -> None:
    rng = np.random.default_rng(42)
    n_sends = 600

    cust_industry = jnp.array([0])
    customer_idx = jnp.zeros(n_sends, dtype=int)

    sig_choices = [0, 6, 8]
    signal_idx_np = rng.choice(sig_choices, size=n_sends)
    opener_idx_np = rng.choice([0, 1, 2], size=n_sends)
    cta_idx_np = np.zeros(n_sends, dtype=int)
    length_idx_np = np.zeros(n_sends, dtype=int)
    age_days_np = rng.uniform(1.0, 10.0, size=n_sends)
    icp_fit_np = rng.uniform(0.3, 0.7, size=n_sends)

    true_betas = {0: 1.8, 6: 0.8, 8: 0.0}
    y_np = np.zeros(n_sends, dtype=float)
    meeting_np = np.zeros(n_sends, dtype=float)

    for i in range(n_sends):
        sig = signal_idx_np[i]
        beta = true_betas[sig]
        decay = np.exp(-age_days_np[i] / 20.0)
        logit_val = -1.5 + beta * decay + 1.0 * (icp_fit_np[i] - 0.5)
        p = 1.0 / (1.0 + np.exp(-logit_val))
        pos = rng.random() < p
        y_np[i] = 1.0 if pos else 0.0
        if pos:
            meeting_np[i] = 1.0 if rng.random() < 0.4 else 0.0

    data = {
        "n_customers": 1,
        "n_industries": 1,
        "n_signals": 9,
        "n_openers": 3,
        "n_ctas": 3,
        "n_lengths": 3,
        "customer_ids": ["cust_smoke"],
        "cust_to_idx": {"cust_smoke": 0},
        "cust_industry": cust_industry,
        "customer_idx": jnp.array(customer_idx),
        "signal_idx": jnp.array(signal_idx_np),
        "opener_idx": jnp.array(opener_idx_np),
        "cta_idx": jnp.array(cta_idx_np),
        "length_idx": jnp.array(length_idx_np),
        "age_days": jnp.array(age_days_np),
        "icp_fit": jnp.array(icp_fit_np),
        "y": jnp.array(y_np),
        "meeting_true": jnp.array(meeting_np),
        "replied_any": jnp.array(y_np),
    }

    fit_res = fit(
        data,
        method="svi",
        svi_steps=1000,
        svi_rank=8,
        num_samples=200,
        seed=42,
    )

    beta_samples = np.asarray(fit_res["samples"]["beta_cs"])
    mean_beta_0 = float(np.mean(beta_samples[:, 0, 0]))
    mean_beta_6 = float(np.mean(beta_samples[:, 0, 6]))
    mean_beta_8 = float(np.mean(beta_samples[:, 0, 8]))

    assert mean_beta_0 > 0.0
    assert mean_beta_0 > mean_beta_6
    assert mean_beta_0 > mean_beta_8
    assert mean_beta_8 == 0.0
