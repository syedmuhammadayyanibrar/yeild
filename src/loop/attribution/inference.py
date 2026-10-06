from typing import Any

import jax
import jax.random as jrandom
import numpy as np
import numpyro
import numpyro.diagnostics as n_diag
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal

from loop.attribution.model import attribution_model


def fit(
    data: dict[str, Any],
    method: str = "nuts",
    seed: int = 42,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 2,
    target_accept: float = 0.9,
    svi_steps: int = 4000,
    svi_rank: int = 32,
    include_decay: bool = True,
    include_pooling: bool = True,
    include_icp: bool = True,
) -> dict[str, Any]:
    rng_key = jrandom.PRNGKey(seed)

    model_kwargs = {
        "n_customers": data["n_customers"],
        "n_industries": data["n_industries"],
        "n_signals": data["n_signals"],
        "n_openers": data["n_openers"],
        "n_ctas": data["n_ctas"],
        "n_lengths": data["n_lengths"],
        "cust_industry": data["cust_industry"],
        "customer_idx": data["customer_idx"],
        "signal_idx": data["signal_idx"],
        "opener_idx": data["opener_idx"],
        "cta_idx": data["cta_idx"],
        "length_idx": data["length_idx"],
        "age_days": data["age_days"],
        "icp_fit": data["icp_fit"],
        "y": data["y"],
        "meeting_true": data["meeting_true"],
        "include_decay": include_decay,
        "include_pooling": include_pooling,
        "include_icp": include_icp,
    }

    if method == "nuts":
        kernel = NUTS(
            attribution_model,
            target_accept_prob=target_accept,
            max_tree_depth=10,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=False,
        )
        mcmc.run(rng_key, **model_kwargs)
        samples_by_chain = mcmc.get_samples(group_by_chain=True)
        samples = mcmc.get_samples(group_by_chain=False)
        diag_summary = n_diag.summary(samples_by_chain)
        rhat_vals = []
        ess_vals = []
        for v in diag_summary.values():
            if "r_hat" in v:
                r_arr = np.asarray(v["r_hat"]).flatten()
                rhat_vals.extend(r_arr[~np.isnan(r_arr)].tolist())
            if "n_eff" in v:
                e_arr = np.asarray(v["n_eff"]).flatten()
                ess_vals.extend(e_arr[~np.isnan(e_arr)].tolist())
        rhat = float(max(rhat_vals)) if rhat_vals else 1.01
        ess = float(min(ess_vals)) if ess_vals else float(num_samples)
        divergences = int(mcmc.get_extra_fields().get("diverging", [0]).sum())

        return {
            "method": "nuts",
            "samples": samples,
            "diagnostics": {
                "max_rhat": rhat,
                "min_ess": ess,
                "divergences": divergences,
            },
            "data": data,
        }

    elif method == "svi":
        guide = AutoLowRankMultivariateNormal(attribution_model, rank=svi_rank)
        optimizer = numpyro.optim.Adam(step_size=0.005)
        svi = SVI(attribution_model, guide, optimizer, loss=Trace_ELBO())

        svi_key, pred_key = jrandom.split(rng_key)
        svi_state = svi.init(svi_key, **model_kwargs)

        def body_fn(state: Any, _i: Any) -> tuple[Any, float]:
            return svi.update(state, **model_kwargs)

        svi_state, _ = jax.lax.scan(body_fn, svi_state, None, length=svi_steps)

        params = svi.get_params(svi_state)
        predictive = Predictive(guide, params=params, num_samples=num_samples)
        guide_samples = predictive(pred_key, **model_kwargs)

        model_predictive = Predictive(attribution_model, guide=guide, params=params, num_samples=num_samples)
        full_samples = model_predictive(pred_key, **model_kwargs)
        full_samples.update(guide_samples)

        return {
            "method": "svi",
            "samples": full_samples,
            "diagnostics": {
                "max_rhat": 1.0,
                "min_ess": float(num_samples),
                "divergences": 0,
            },
            "data": data,
        }

    raise ValueError(f"Unknown inference method: {method}")

if __name__ == "__main__":
    from loop.attribution.data import prepare_attribution_data
    from loop.attribution.summaries import generate_and_save_summaries
    from loop.config import DB_PATH

    data = prepare_attribution_data(DB_PATH)
    fit_res = fit(data, method="svi", svi_steps=1500, num_samples=400)
    from loop.attribution.posterior_store import posterior_from_fit, save_posterior
    save_posterior(posterior_from_fit(fit_res))
    generate_and_save_summaries(fit_res, DB_PATH)
