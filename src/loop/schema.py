from pathlib import Path
from typing import Literal

import duckdb

Industry = Literal[
    "saas_devtools",
    "ecommerce_dtc",
    "fintech",
    "marketing_agencies",
    "industrial_b2b",
]

INDUSTRIES: tuple[Industry, ...] = (
    "saas_devtools",
    "ecommerce_dtc",
    "fintech",
    "marketing_agencies",
    "industrial_b2b",
)

SignalType = Literal[
    "pain_point_post",
    "competitor_engagement",
    "job_change_into_role",
    "funding_announced",
    "leadership_hire",
    "hiring_for_target_role",
    "event_attendance",
    "tech_stack_change",
    "no_signal",
]

SIGNAL_TYPES: tuple[SignalType, ...] = (
    "pain_point_post",
    "competitor_engagement",
    "job_change_into_role",
    "funding_announced",
    "leadership_hire",
    "hiring_for_target_role",
    "event_attendance",
    "tech_stack_change",
    "no_signal",
)

OpenerType = Literal["trigger_specific", "question_led", "generic_value"]
OPENER_TYPES: tuple[OpenerType, ...] = ("trigger_specific", "question_led", "generic_value")

CtaType = Literal["soft_question", "resource_offer", "calendar_link"]
CTA_TYPES: tuple[CtaType, ...] = ("soft_question", "resource_offer", "calendar_link")

LengthBucket = Literal["short", "medium", "long"]
LENGTH_BUCKETS: tuple[LengthBucket, ...] = ("short", "medium", "long")

Language = Literal["en", "fr", "de", "es"]
LANGUAGES: tuple[Language, ...] = ("en", "fr", "de", "es")

ReplyClass = Literal[
    "interested",
    "info_request",
    "referral",
    "objection_timing",
    "objection_budget",
    "objection_competitor",
    "not_relevant",
    "wrong_person",
    "out_of_office",
    "unsubscribe",
]

REPLY_CLASSES: tuple[ReplyClass, ...] = (
    "interested",
    "info_request",
    "referral",
    "objection_timing",
    "objection_budget",
    "objection_competitor",
    "not_relevant",
    "wrong_person",
    "out_of_office",
    "unsubscribe",
)

POSITIVE_CLASSES: frozenset[ReplyClass] = frozenset({"interested", "info_request"})

ALL_ARMS: tuple[str, ...] = tuple(
    f"{sig}:{op}" for sig in SIGNAL_TYPES for op in OPENER_TYPES
)

def init_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    con.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        industry VARCHAR NOT NULL,
        primary_language VARCHAR NOT NULL,
        n_sends INTEGER NOT NULL
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS sends (
        send_id VARCHAR PRIMARY KEY,
        customer_id VARCHAR NOT NULL,
        week INTEGER NOT NULL,
        signal_type VARCHAR NOT NULL,
        signal_age_days DOUBLE NOT NULL,
        icp_fit DOUBLE NOT NULL,
        opener_type VARCHAR NOT NULL,
        cta_type VARCHAR NOT NULL,
        length_bucket VARCHAR NOT NULL,
        language VARCHAR NOT NULL,
        arm_id VARCHAR NOT NULL,
        policy VARCHAR NOT NULL,
        propensity DOUBLE NOT NULL,
        replied_any BOOLEAN NOT NULL,
        positive_true BOOLEAN NOT NULL,
        meeting_true BOOLEAN NOT NULL
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS replies (
        reply_id VARCHAR PRIMARY KEY,
        send_id VARCHAR NOT NULL,
        customer_id VARCHAR NOT NULL,
        received_at VARCHAR NOT NULL,
        text VARCHAR NOT NULL,
        true_class VARCHAR NOT NULL,
        true_language VARCHAR NOT NULL,
        referral_name VARCHAR,
        referral_email VARCHAR,
        revisit_window_start VARCHAR,
        revisit_window_end VARCHAR,
        split VARCHAR NOT NULL
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS classifications (
        reply_id VARCHAR PRIMARY KEY,
        model VARCHAR NOT NULL,
        class_probs_json VARCHAR NOT NULL,
        predicted_class VARCHAR NOT NULL,
        confidence DOUBLE NOT NULL,
        routed_to_review BOOLEAN NOT NULL,
        language_detected VARCHAR NOT NULL,
        referral_name VARCHAR,
        referral_email VARCHAR,
        revisit_date VARCHAR,
        draft_reply VARCHAR,
        validators_json VARCHAR NOT NULL
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS posterior_summaries (
        customer_id VARCHAR NOT NULL,
        quantity VARCHAR NOT NULL,
        key VARCHAR NOT NULL,
        median DOUBLE NOT NULL,
        lo90 DOUBLE NOT NULL,
        hi90 DOUBLE NOT NULL,
        extra_json VARCHAR,
        PRIMARY KEY (customer_id, quantity, key)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS briefs (
        customer_id VARCHAR PRIMARY KEY,
        brief_json VARCHAR NOT NULL
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS actions (
        reply_id VARCHAR PRIMARY KEY,
        customer_id VARCHAR NOT NULL,
        action VARCHAR NOT NULL,
        final_text VARCHAR NOT NULL,
        corrected_class VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS suppression_list (
        email_or_domain VARCHAR PRIMARY KEY,
        customer_id VARCHAR NOT NULL,
        reason VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS followups (
        reply_id VARCHAR PRIMARY KEY,
        customer_id VARCHAR NOT NULL,
        contact_hint VARCHAR,
        scheduled_date VARCHAR NOT NULL,
        status VARCHAR NOT NULL
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS referral_leads (
        lead_id VARCHAR PRIMARY KEY,
        customer_id VARCHAR NOT NULL,
        name VARCHAR,
        email VARCHAR NOT NULL,
        source_reply_id VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS lead_pool (
        lead_id VARCHAR PRIMARY KEY,
        customer_id VARCHAR NOT NULL,
        company_name VARCHAR NOT NULL,
        title VARCHAR NOT NULL,
        signal_type VARCHAR NOT NULL,
        signal_age_days DOUBLE NOT NULL,
        icp_fit DOUBLE NOT NULL,
        language VARCHAR NOT NULL,
        is_sent BOOLEAN NOT NULL,
        allocated_arm VARCHAR,
        send_id VARCHAR
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS weekly_ledger (
        week INTEGER NOT NULL,
        customer_id VARCHAR NOT NULL,
        policy VARCHAR NOT NULL,
        sends INTEGER NOT NULL,
        positive_replies INTEGER NOT NULL,
        meetings INTEGER NOT NULL,
        counterfactual_meetings DOUBLE NOT NULL,
        difference_meetings DOUBLE NOT NULL,
        PRIMARY KEY (customer_id, week)
    );
    """)

    return con

