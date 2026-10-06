from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from loop.attribution.data import CTA_TO_IDX, LENGTH_TO_IDX, OPENER_TO_IDX, SIGNAL_TO_IDX
from loop.attribution.posterior_store import Posterior
from loop.schema import OPENER_TYPES

DEFAULT_CTA = "soft_question"
DEFAULT_LENGTH = "short"


@dataclass(frozen=True)
class LeadArrays:
    signal_idx: np.ndarray
    age_days: np.ndarray
    icp_fit: np.ndarray


@dataclass(frozen=True)
class CustomerContext:
    customer_idx: int
    industry_idx: int


def lead_arrays(signal_types: list[str], ages: list[float], icp: list[float]) -> LeadArrays:
    return LeadArrays(
        signal_idx=np.array([SIGNAL_TO_IDX[s] for s in signal_types], dtype=int),
        age_days=np.asarray(ages, dtype=float),
        icp_fit=np.asarray(icp, dtype=float),
    )


def logit_terms(post: Posterior, ctx: CustomerContext, leads: LeadArrays, opener_idx: int) -> dict[str, np.ndarray]:
    d = post.draws
    c, k = ctx.customer_idx, ctx.industry_idx
    base = (
        d["alpha_c"][:, c]
        + d["eta"][:, k, CTA_TO_IDX[DEFAULT_CTA]]
        + d["lambda_len"][:, LENGTH_TO_IDX[DEFAULT_LENGTH]]
    )[:, None] * np.ones((1, leads.signal_idx.shape[0]))
    icp_term = d["gamma"][:, None] * (leads.icp_fit[None, :] - 0.5)
    decay = np.exp(-leads.age_days[None, :] / d["tau_s"][:, leads.signal_idx])
    signal_term = d["beta_cs"][:, c, :][:, leads.signal_idx] * decay
    opener_term = np.broadcast_to(d["delta"][:, k, opener_idx][:, None], base.shape)
    return {"base": base, "icp": icp_term, "signal": signal_term, "opener": opener_term}


def meeting_prob_draws(post: Posterior, ctx: CustomerContext, leads: LeadArrays) -> np.ndarray:
    p_meet_given_pos = expit(post.draws["a2_c"][:, ctx.customer_idx])[:, None]
    per_opener = []
    for op in OPENER_TYPES:
        terms = logit_terms(post, ctx, leads, OPENER_TO_IDX[op])
        logit = terms["base"] + terms["icp"] + terms["signal"] + terms["opener"]
        per_opener.append(expit(logit) * p_meet_given_pos)
    return np.stack(per_opener, axis=2)


def positive_prob_draws_for_choice(
    post: Posterior, ctx: CustomerContext, leads: LeadArrays, opener_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    stacked = []
    for op_i in range(len(OPENER_TYPES)):
        terms = logit_terms(post, ctx, leads, op_i)
        stacked.append(terms["base"] + terms["icp"] + terms["signal"] + terms["opener"])
    logits = np.stack(stacked, axis=2)
    chosen = np.take_along_axis(logits, opener_idx[None, :, None], axis=2)[:, :, 0]
    p_meet_given_pos = expit(post.draws["a2_c"][:, ctx.customer_idx])
    return expit(chosen), p_meet_given_pos


def signal_retention(post: Posterior, signal_idx: int, age_days: float) -> float:
    tau = post.draws["tau_s"][:, signal_idx]
    return float(np.sum(post.weights * np.exp(-age_days / tau)))


def signal_half_life_days(post: Posterior, signal_idx: int) -> float:
    tau = post.draws["tau_s"][:, signal_idx]
    return float(np.sum(post.weights * tau) * np.log(2.0))


def explain_lead(
    post: Posterior, ctx: CustomerContext, leads: LeadArrays, lead_i: int, opener_idx: int
) -> dict[str, float]:
    single = LeadArrays(
        signal_idx=leads.signal_idx[lead_i : lead_i + 1],
        age_days=leads.age_days[lead_i : lead_i + 1],
        icp_fit=leads.icp_fit[lead_i : lead_i + 1],
    )
    terms = logit_terms(post, ctx, single, opener_idx)
    w = post.weights
    return {name: float(np.sum(w * arr[:, 0])) for name, arr in terms.items()}
