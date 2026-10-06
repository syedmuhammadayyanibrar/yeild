from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"

class CustomerItem(BaseModel):
    customer_id: str
    name: str
    industry: str
    primary_language: str
    n_sends: int

class CustomerListResponse(BaseModel):
    customers: list[CustomerItem]

class BriefResponse(BaseModel):
    customer_id: str
    customer_name: str
    industry: str
    primary_language: str
    window: dict[str, Any]
    kpi_strip: dict[str, Any]
    allocation_table: list[dict[str, Any]]
    stop_list: list[dict[str, Any]]
    test_list: list[dict[str, Any]]

class SignalItem(BaseModel):
    signal: str
    median_lift: float
    lo90: float
    hi90: float
    half_life_median: float
    half_life_lo90: float
    half_life_hi90: float
    naive_lift: float
    decay_curve: list[dict[str, float]]

class CustomerSignalsResponse(BaseModel):
    customer_id: str
    customer_name: str
    industry: str
    finding_title: str
    signals: list[SignalItem]

class ReplyListItem(BaseModel):
    reply_id: str
    send_id: str
    customer_id: str
    received_at: str
    text: str
    true_class: str
    true_language: str
    predicted_class: str
    confidence: float
    routed_to_review: bool
    referral_name: str | None = None
    referral_email: str | None = None
    revisit_date: str | None = None
    draft_reply: str | None = None
    validators: dict[str, bool] = Field(default_factory=dict)
    class_probabilities: dict[str, float] = Field(default_factory=dict)

class RepliesResponse(BaseModel):
    total: int
    limit: int
    offset: int
    auto_handled_count: int
    review_count: int
    replies: list[ReplyListItem]

class ClassifyRequest(BaseModel):
    text: str
    received_at: str | None = None

class ClassifyResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    language: str
    referral_name: str | None = None
    referral_email: str | None = None
    revisit_date: str | None = None
    draft_reply: str | None = None
    validators: dict[str, bool] = Field(default_factory=dict)
    routed_to_review: bool = False
    model: str

class InboxItem(BaseModel):
    reply_id: str
    send_id: str
    customer_id: str
    received_at: str
    text: str
    predicted_class: str
    confidence: float
    routed_to_review: bool
    language: str
    draft_reply: str | None = None
    referral_name: str | None = None
    referral_email: str | None = None
    revisit_date: str | None = None
    category: str
    action_taken: str | None = None

class InboxResponse(BaseModel):
    total: int
    needs_you_count: int
    ready_to_send_count: int
    done_automatically_count: int
    needs_you: list[InboxItem]
    ready_to_send: list[InboxItem]
    done_automatically: list[InboxItem]

class InboxActionRequest(BaseModel):
    action: str
    final_text: str | None = None
    corrected_class: str | None = None

class LeadItem(BaseModel):
    lead_id: str
    company_name: str
    title: str
    signal_type: str
    signal_age_days: float
    icp_fit: float
    opener_type: str
    expected_meetings_per_100: float
    log_odds_terms: dict[str, float] = Field(default_factory=dict)
    stale_by_friday: bool
    why: str

class LeadsTodayResponse(BaseModel):
    customer_id: str
    total_available: int
    leads: list[LeadItem]

class PlanRecommendationItem(BaseModel):
    action: str
    arm_id: str
    signal_type: str
    opener_type: str
    current_share_pct: float
    recommended_share_pct: float
    expected_meetings: float
    lo90: float
    hi90: float
    sends_observed: int
    rationale: str

class PlanResponse(BaseModel):
    customer_id: str
    current_expected_meetings: float
    current_lo90: float
    current_hi90: float
    recommended_expected_meetings: float
    recommended_lo90: float
    recommended_hi90: float
    relative_gain_pct: float
    p_recommended_beats_current: float
    recommendations: list[PlanRecommendationItem]

class ApplyPlanRequest(BaseModel):
    shares: dict[str, float] | None = None

class LedgerRowItem(BaseModel):
    week: int
    plan: str
    sends: int
    positive_replies: int
    meetings: int
    counterfactual_meetings: float
    difference_meetings: float

class ModelGateCheck(BaseModel):
    gate_id: str
    description: str
    status: str
    value: str

class LedgerResponse(BaseModel):
    customer_id: str
    rows: list[LedgerRowItem]
    total_meetings_loop: int
    total_meetings_counterfactual: float
    net_gain_meetings: float
    gates: list[ModelGateCheck]

class RunWeekResponse(BaseModel):
    success: bool
    new_row: LedgerRowItem
    message: str

