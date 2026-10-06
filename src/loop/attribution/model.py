import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import expit
from scipy.special import logit


def attribution_model(
    n_customers: int,
    n_industries: int,
    n_signals: int,
    n_openers: int,
    n_ctas: int,
    n_lengths: int,
    cust_industry: jnp.ndarray,
    customer_idx: jnp.ndarray,
    signal_idx: jnp.ndarray,
    opener_idx: jnp.ndarray,
    cta_idx: jnp.ndarray,
    length_idx: jnp.ndarray,
    age_days: jnp.ndarray,
    icp_fit: jnp.ndarray,
    y: jnp.ndarray,
    meeting_true: jnp.ndarray,
    include_decay: bool = True,
    include_pooling: bool = True,
    include_icp: bool = True,
) -> None:
    logit_003 = float(logit(0.03))
    logit_045 = float(logit(0.45))

    alpha0 = numpyro.sample("alpha0", dist.Normal(logit_003, 0.5))
    sigma_alpha_k = numpyro.sample("sigma_alpha_k", dist.HalfNormal(0.35))
    alpha_k_offset = numpyro.sample("alpha_k_offset", dist.Normal(0.0, 1.0).expand([n_industries]))
    alpha_k = alpha0 + sigma_alpha_k * alpha_k_offset

    sigma_alpha_c = numpyro.sample("sigma_alpha_c", dist.HalfNormal(0.30))
    alpha_c_offset = numpyro.sample("alpha_c_offset", dist.Normal(0.0, 1.0).expand([n_customers]))
    alpha_c = numpyro.deterministic("alpha_c", alpha_k[cust_industry] + sigma_alpha_c * alpha_c_offset)

    if include_icp:
        gamma = numpyro.sample("gamma", dist.Normal(1.0, 0.5))
    else:
        gamma = 0.0

    mu0_active = numpyro.sample("mu0_active", dist.Normal(0.3, 0.5).expand([n_signals - 1]))
    mu0_s = jnp.concatenate([mu0_active, jnp.zeros(1)])

    if include_pooling:
        sigma_ind_active = numpyro.sample("sigma_ind_active", dist.HalfNormal(0.3).expand([n_signals - 1]))
        sigma_industry_s = jnp.concatenate([sigma_ind_active, jnp.zeros(1)])
        mu_ks_offset = numpyro.sample("mu_ks_offset", dist.Normal(0.0, 1.0).expand([n_industries, n_signals]))
        mu_ks = mu0_s + sigma_industry_s * mu_ks_offset
    else:
        mu_ks = jnp.repeat(mu0_s[None, :], n_industries, axis=0)

    sigma_customer = numpyro.sample("sigma_customer", dist.HalfNormal(0.25))
    beta_cs_offset = numpyro.sample("beta_cs_offset", dist.Normal(0.0, 1.0).expand([n_customers, n_signals]))

    mu_for_cust = mu_ks[cust_industry]
    mask_active = jnp.ones(n_signals).at[-1].set(0.0)
    beta_cs = numpyro.deterministic("beta_cs", (mu_for_cust + sigma_customer * beta_cs_offset) * mask_active)

    if include_decay:
        tau_active = numpyro.sample("tau_active", dist.LogNormal(jnp.log(30.0), 0.7).expand([n_signals - 1]))
        tau_s = numpyro.deterministic("tau_s", jnp.concatenate([tau_active, jnp.array([1e9])]))
    else:
        tau_s = jnp.full(n_signals, 1e9)

    delta = numpyro.sample("delta", dist.Normal(0.0, 0.4).expand([n_industries, n_openers]))
    eta = numpyro.sample("eta", dist.Normal(0.0, 0.4).expand([n_industries, n_ctas]))
    lambda_len = numpyro.sample("lambda_len", dist.Normal(0.0, 0.4).expand([n_lengths]))

    cust_ind = cust_industry[customer_idx]
    decay_factors = jnp.exp(-age_days / tau_s[signal_idx])
    sig_effects = beta_cs[customer_idx, signal_idx] * decay_factors

    logit_p = (
        alpha_c[customer_idx]
        + gamma * (icp_fit - 0.5)
        + sig_effects
        + delta[cust_ind, opener_idx]
        + eta[cust_ind, cta_idx]
        + lambda_len[length_idx]
    )

    p_pos = expit(logit_p)
    eps = 1e-12
    p_safe = jnp.clip(p_pos, eps, 1.0 - eps)

    soft_log_lik = y * jnp.log(p_safe) + (1.0 - y) * jnp.log(1.0 - p_safe)
    numpyro.factor("soft_log_lik", jnp.sum(soft_log_lik))

    a20 = numpyro.sample("a20", dist.Normal(logit_045, 0.5))
    sigma_a2_k = numpyro.sample("sigma_a2_k", dist.HalfNormal(0.30))
    a2_k_offset = numpyro.sample("a2_k_offset", dist.Normal(0.0, 1.0).expand([n_industries]))
    a2_k = a20 + sigma_a2_k * a2_k_offset

    sigma_a2_c = numpyro.sample("sigma_a2_c", dist.HalfNormal(0.25))
    a2_c_offset = numpyro.sample("a2_c_offset", dist.Normal(0.0, 1.0).expand([n_customers]))
    a2_c = numpyro.deterministic("a2_c", a2_k[cust_industry] + sigma_a2_c * a2_c_offset)

    p_meet_pos = expit(a2_c[customer_idx])
    p_meet_safe = jnp.clip(p_meet_pos, eps, 1.0 - eps)
    meeting_log_lik = meeting_true * jnp.log(p_meet_safe) + (1.0 - meeting_true) * jnp.log(1.0 - p_meet_safe)
    numpyro.factor("meeting_log_lik", jnp.sum(y * meeting_log_lik))
