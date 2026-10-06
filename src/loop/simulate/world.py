from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit, logit

from loop.schema import (
    ALL_ARMS,
    CTA_TYPES,
    INDUSTRIES,
    LANGUAGES,
    OPENER_TYPES,
    SIGNAL_TYPES,
    CtaType,
    Industry,
    Language,
    LengthBucket,
    OpenerType,
    ReplyClass,
    SignalType,
)

SIGNAL_CONFIGS: dict[SignalType, tuple[float, float, float]] = {
    "pain_point_post": (0.80, 7.0, 5.0),
    "competitor_engagement": (0.60, 10.0, 6.0),
    "job_change_into_role": (0.70, 45.0, 25.0),
    "funding_announced": (0.55, 21.0, 14.0),
    "leadership_hire": (0.40, 30.0, 20.0),
    "hiring_for_target_role": (0.35, 60.0, 35.0),
    "event_attendance": (0.20, 14.0, 9.0),
    "tech_stack_change": (0.30, 90.0, 45.0),
    "no_signal": (0.0, 1e9, 0.0),
}

OPENER_EFFECTS: dict[OpenerType, float] = {
    "trigger_specific": 0.35,
    "question_led": 0.10,
    "generic_value": 0.0,
}

CTA_EFFECTS: dict[CtaType, float] = {
    "soft_question": 0.15,
    "resource_offer": 0.05,
    "calendar_link": 0.0,
}

LENGTH_EFFECTS: dict[LengthBucket, float] = {
    "short": 0.10,
    "medium": 0.0,
    "long": -0.20,
}

NON_POSITIVE_CLASS_MIX_BY_INDUSTRY: dict[Industry, dict[ReplyClass, float]] = {
    "saas_devtools": {
        "out_of_office": 0.25,
        "unsubscribe": 0.15,
        "not_relevant": 0.20,
        "objection_competitor": 0.15,
        "objection_budget": 0.08,
        "objection_timing": 0.07,
        "wrong_person": 0.05,
        "referral": 0.05,
    },
    "ecommerce_dtc": {
        "out_of_office": 0.20,
        "unsubscribe": 0.25,
        "not_relevant": 0.22,
        "objection_budget": 0.12,
        "objection_competitor": 0.08,
        "objection_timing": 0.05,
        "wrong_person": 0.04,
        "referral": 0.04,
    },
    "fintech": {
        "out_of_office": 0.22,
        "unsubscribe": 0.18,
        "not_relevant": 0.18,
        "objection_competitor": 0.14,
        "objection_budget": 0.10,
        "objection_timing": 0.08,
        "wrong_person": 0.05,
        "referral": 0.05,
    },
    "marketing_agencies": {
        "out_of_office": 0.18,
        "unsubscribe": 0.20,
        "not_relevant": 0.15,
        "objection_budget": 0.18,
        "objection_timing": 0.12,
        "objection_competitor": 0.07,
        "wrong_person": 0.05,
        "referral": 0.05,
    },
    "industrial_b2b": {
        "out_of_office": 0.25,
        "unsubscribe": 0.15,
        "not_relevant": 0.20,
        "wrong_person": 0.12,
        "referral": 0.10,
        "objection_budget": 0.08,
        "objection_timing": 0.06,
        "objection_competitor": 0.04,
    },
}

@dataclass
class CustomerProfile:
    customer_id: str
    name: str
    industry: Industry
    primary_language: Language
    n_sends: int
    alpha_c: float
    beta_cs: dict[SignalType, float]
    delta_opener: dict[OpenerType, float]
    eta_cta: dict[CtaType, float]
    lambda_len: dict[LengthBucket, float]
    meeting_logit: float
    p_non_positive_reply: float
    dominant_opener: OpenerType
    signal_preference_logits: dict[SignalType, float]
    default_cta: CtaType
    default_length: LengthBucket
    ground_truth_arm_values: dict[str, float] = field(default_factory=dict)

@dataclass
class World:
    seed: int
    alpha_k: dict[Industry, float]
    mu_ks: dict[tuple[Industry, SignalType], float]
    opener_k: dict[tuple[Industry, OpenerType], float]
    cta_k: dict[tuple[Industry, CtaType], float]
    meeting_k: dict[Industry, float]
    customers: dict[str, CustomerProfile]

def sample_signal_age(signal_type: SignalType, rng: np.random.Generator) -> float:
    if signal_type == "no_signal":
        return 0.0
    _, _, mean_age = SIGNAL_CONFIGS[signal_type]
    raw_age = float(rng.exponential(mean_age))
    return min(90.0, raw_age)

def generate_world(seed: int, mc_leads: int = 200000) -> World:
    rng = np.random.default_rng(seed)

    alpha_k: dict[Industry, float] = {}
    mu_ks: dict[tuple[Industry, SignalType], float] = {}
    opener_k: dict[tuple[Industry, OpenerType], float] = {}
    cta_k: dict[tuple[Industry, CtaType], float] = {}
    meeting_k: dict[Industry, float] = {}

    logit_003 = float(logit(0.03))
    logit_045 = float(logit(0.45))

    for ind in INDUSTRIES:
        alpha_k[ind] = float(rng.normal(logit_003, 0.35))
        meeting_k[ind] = float(rng.normal(logit_045, 0.30))
        for sig in SIGNAL_TYPES:
            global_eff, _, _ = SIGNAL_CONFIGS[sig]
            mu_ks[(ind, sig)] = float(global_eff + (rng.normal(0.0, 0.25) if sig != "no_signal" else 0.0))
        for op in OPENER_TYPES:
            opener_k[(ind, op)] = float(OPENER_EFFECTS[op] + rng.normal(0.0, 0.15))
        for cta in CTA_TYPES:
            cta_k[(ind, cta)] = float(CTA_EFFECTS[cta] + rng.normal(0.0, 0.15))

    customers: dict[str, CustomerProfile] = {}
    cust_counter = 0

    for ind in INDUSTRIES:
        for _ in range(8):
            cust_counter += 1
            cid = f"cust_{cust_counter:03d}"
            name = f"{ind.replace('_', ' ').title()} Co {cust_counter}"
            prim_lang: Language = LANGUAGES[int(rng.integers(0, len(LANGUAGES)))]

            raw_sends = float(np.exp(rng.normal(np.log(1100), 0.65)))
            n_sends = int(np.clip(raw_sends, 300, 4000))

            alpha_c = float(rng.normal(alpha_k[ind], 0.30))
            beta_cs: dict[SignalType, float] = {}
            for sig in SIGNAL_TYPES:
                if sig == "no_signal":
                    beta_cs[sig] = 0.0
                else:
                    beta_cs[sig] = float(rng.normal(mu_ks[(ind, sig)], 0.20))

            delta_opener: dict[OpenerType, float] = {}
            for op in OPENER_TYPES:
                delta_opener[op] = opener_k[(ind, op)]

            eta_cta: dict[CtaType, float] = {}
            for cta in CTA_TYPES:
                eta_cta[cta] = cta_k[(ind, cta)]

            lambda_len: dict[LengthBucket, float] = {
                "short": LENGTH_EFFECTS["short"],
                "medium": LENGTH_EFFECTS["medium"],
                "long": LENGTH_EFFECTS["long"],
            }

            meeting_logit = float(rng.normal(meeting_k[ind], 0.25))
            p_non_pos = float(rng.uniform(0.02, 0.04))

            dominant_opener: OpenerType = OPENER_TYPES[int(rng.integers(0, len(OPENER_TYPES)))]
            sig_pref_logits: dict[SignalType, float] = {}
            for sig in SIGNAL_TYPES:
                sig_pref_logits[sig] = float(rng.normal(0.0, 0.5))

            default_cta: CtaType = "soft_question"
            default_length: LengthBucket = "short"

            profile = CustomerProfile(
                customer_id=cid,
                name=name,
                industry=ind,
                primary_language=prim_lang,
                n_sends=n_sends,
                alpha_c=alpha_c,
                beta_cs=beta_cs,
                delta_opener=delta_opener,
                eta_cta=eta_cta,
                lambda_len=lambda_len,
                meeting_logit=meeting_logit,
                p_non_positive_reply=p_non_pos,
                dominant_opener=dominant_opener,
                signal_preference_logits=sig_pref_logits,
                default_cta=default_cta,
                default_length=default_length,
            )
            customers[cid] = profile

    world = World(
        seed=seed,
        alpha_k=alpha_k,
        mu_ks=mu_ks,
        opener_k=opener_k,
        cta_k=cta_k,
        meeting_k=meeting_k,
        customers=customers,
    )

    compute_ground_truth_arm_values(world, mc_leads=mc_leads, seed=12345)
    return world

def compute_single_send_positive_prob(
    profile: CustomerProfile,
    signal_type: SignalType,
    age_days: float,
    icp_fit: float,
    opener_type: OpenerType,
    cta_type: CtaType,
    length_bucket: LengthBucket,
) -> float:
    _, tau, _ = SIGNAL_CONFIGS[signal_type]
    decay = np.exp(-age_days / tau) if tau > 0 else 0.0
    beta = profile.beta_cs[signal_type]

    logit_p = (
        profile.alpha_c
        + 1.2 * (icp_fit - 0.5)
        + beta * decay
        + profile.delta_opener[opener_type]
        + profile.eta_cta[cta_type]
        + profile.lambda_len[length_bucket]
    )
    return float(expit(logit_p))

def compute_single_send_meeting_prob(
    profile: CustomerProfile,
    signal_type: SignalType,
    age_days: float,
    icp_fit: float,
    opener_type: OpenerType,
    cta_type: CtaType,
    length_bucket: LengthBucket,
) -> float:
    p_pos = compute_single_send_positive_prob(
        profile, signal_type, age_days, icp_fit, opener_type, cta_type, length_bucket
    )
    p_meet_given_pos = float(expit(profile.meeting_logit))
    return p_pos * p_meet_given_pos

def compute_ground_truth_arm_values(
    world: World,
    mc_leads: int = 200000,
    seed: int = 12345,
) -> None:
    mc_rng = np.random.default_rng(seed)

    precomputed_samples: dict[SignalType, tuple[np.ndarray, np.ndarray]] = {}
    for sig in SIGNAL_TYPES:
        _, _, mean_age = SIGNAL_CONFIGS[sig]
        if sig == "no_signal":
            ages = np.zeros(mc_leads, dtype=float)
        else:
            ages = np.minimum(90.0, mc_rng.exponential(mean_age, size=mc_leads))
        icp = mc_rng.beta(3.0, 3.0, size=mc_leads)
        precomputed_samples[sig] = (ages, icp)

    for profile in world.customers.values():
        p_meet_given_pos = float(expit(profile.meeting_logit))
        for arm_id in ALL_ARMS:
            sig_str, op_str = arm_id.split(":")
            sig: SignalType = sig_str
            op: OpenerType = op_str
            ages, icp = precomputed_samples[sig]
            _, tau, _ = SIGNAL_CONFIGS[sig]
            decay = np.exp(-ages / tau)
            beta = profile.beta_cs[sig]

            logits = (
                profile.alpha_c
                + 1.2 * (icp - 0.5)
                + beta * decay
                + profile.delta_opener[op]
                + profile.eta_cta[profile.default_cta]
                + profile.lambda_len[profile.default_length]
            )
            p_pos_samples = expit(logits)
            expected_meetings_per_send = float(np.mean(p_pos_samples) * p_meet_given_pos)
            profile.ground_truth_arm_values[arm_id] = expected_meetings_per_send
