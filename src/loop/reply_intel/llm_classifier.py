from typing import Any

import numpy as np

from loop.reply_intel.llm_client import LlmClient
from loop.reply_intel.prompts import (
    CLASSIFY_REPLY_TOOL_SCHEMA,
    SYSTEM_PROMPT,
    format_user_prompt,
)
from loop.reply_intel.taxonomy import INDEX_TO_CLASS
from loop.schema import REPLY_CLASSES


def classify_with_llm(
    text: str,
    received_at: str | None = None,
    client: LlmClient | None = None,
) -> dict[str, Any] | None:
    if client is None:
        client = LlmClient()

    if not client.is_available():
        return None

    user_content = format_user_prompt(text, received_at)
    res = client.call_cached(
        system_prompt=SYSTEM_PROMPT,
        user_content=user_content,
        tool_schema=CLASSIFY_REPLY_TOOL_SCHEMA,
    )

    if res is None or not res.get("structured"):
        return None

    data = res["structured"]
    probs_dict = data.get("probabilities", {})

    prob_vec = np.zeros(len(REPLY_CLASSES), dtype=float)
    for idx, cls in enumerate(REPLY_CLASSES):
        prob_vec[idx] = float(probs_dict.get(cls, 0.0))

    if prob_vec.sum() > 0:
        prob_vec /= prob_vec.sum()
    else:
        prob_vec = np.ones(len(REPLY_CLASSES), dtype=float) / len(REPLY_CLASSES)

    best_idx = int(np.argmax(prob_vec))
    pred_cls = INDEX_TO_CLASS[best_idx]
    confidence = float(prob_vec[best_idx])

    probs_normalized = {REPLY_CLASSES[i]: float(prob_vec[i]) for i in range(len(REPLY_CLASSES))}

    return {
        "model": client.model,
        "probabilities": probs_normalized,
        "prob_vector": prob_vec,
        "predicted_class": pred_cls,
        "confidence": confidence,
        "language": data.get("language", "en"),
        "referral_name": data.get("referral_name"),
        "referral_email": data.get("referral_email"),
        "revisit_date": data.get("revisit_date"),
        "input_tokens": res.get("input_tokens", 0),
        "output_tokens": res.get("output_tokens", 0),
        "latency_ms": res.get("latency_ms", 0.0),
    }
