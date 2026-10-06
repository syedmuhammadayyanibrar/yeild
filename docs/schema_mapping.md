# Production Funnel Schema Mapping

This document maps production go-to-market funnel entities and fields onto Loop's DuckDB tables.

## 1. Funnel Stages and Entity Mapping

| Funnel Entity | Production Source Concepts | Loop Target Table | Key Mapped Attributes |
| :--- | :--- | :--- | :--- |
| Customer / Account | Outbound sender account, tenant, client ID | `customers` | `customer_id`, `name`, `industry`, `primary_language`, `n_sends` |
| Signal | Buying signal detection, intent event, job change | `sends` | `signal_type`, `signal_age_days`, `icp_fit` |
| Outbound Send | Sent email, sequence touchpoint, campaign dispatch | `sends` | `send_id`, `customer_id`, `week`, `opener_type`, `cta_type`, `length_bucket`, `language`, `arm_id`, `policy`, `propensity` |
| Reply Event | Inbound message, webhook payload, thread reply | `replies` | `reply_id`, `send_id`, `customer_id`, `received_at`, `text`, `true_class`, `true_language`, `referral_name`, `referral_email`, `revisit_window_start`, `revisit_window_end`, `split` |
| Intelligence & Routing | LLM / baseline classification, routing decision | `classifications` | `reply_id`, `model`, `class_probs_json`, `predicted_class`, `confidence`, `routed_to_review`, `language_detected`, `referral_name`, `referral_email`, `revisit_date`, `draft_reply`, `validators_json` |
| Meeting / Conversion | Calendly webhook, CRM opportunity stage | `sends` / `posterior_summaries` | `replied_any`, `positive_true`, `meeting_true`, `posterior_summaries` (expected meetings / 100 sends) |
| Recommendation | Campaign brief, allocation update | `briefs` | `customer_id`, `brief_json` |

## 2. Detailed Attribute Translation

### Outbound Send Attribution Features
- `signal_type`: The primary trigger detected prior to dispatch (e.g. `pain_point_post`, `job_change_into_role`, `no_signal`).
- `signal_age_days`: Time elapsed from signal observation timestamp to send timestamp.
- `icp_fit`: Model-scored fit score in `[0.0, 1.0]`.
- `opener_type`: Classification of opening sentence structure (`trigger_specific`, `question_led`, `generic_value`).
- `cta_type`: Call to action format (`soft_question`, `resource_offer`, `calendar_link`).
- `length_bucket`: Word count category (`short`: <60 words, `medium`: 60-120 words, `long`: >120 words).
- `arm_id`: Unique identifier of the experimental arm (`signal_type:opener_type`).
- `propensity`: Probability that the logging policy selected `arm_id` for this lead profile.

### Reply Intelligence Attributes
- `class_probs_json`: Soft probability distribution over the 10 taxonomy classes.
- `confidence`: Maximum class probability, post-temperature calibration.
- `routed_to_review`: Boolean indicating whether confidence fell below the auto-handling threshold `tau`.
- `revisit_date`: Extracted ISO date representing earliest follow-up date for timing objections.
- `validators_json`: Structured results of deterministic guardrail checks evaluated on generated drafts.
