from typing import cast

from loop.schema import (
    POSITIVE_CLASSES,
    REPLY_CLASSES,
    ReplyClass,
)

PRECEDENCE_ORDER: tuple[ReplyClass, ...] = (
    "out_of_office",
    "unsubscribe",
    "referral",
    "wrong_person",
    "objection_timing",
    "objection_budget",
    "objection_competitor",
    "not_relevant",
    "info_request",
    "interested",
)

CLASS_TO_INDEX: dict[ReplyClass, int] = {cls: idx for idx, cls in enumerate(REPLY_CLASSES)}
INDEX_TO_CLASS: dict[int, ReplyClass] = {idx: cls for idx, cls in enumerate(REPLY_CLASSES)}

def is_positive_class(cls: ReplyClass | str) -> bool:
    return cls in POSITIVE_CLASSES

def resolve_precedence(classes: list[ReplyClass | str]) -> ReplyClass:
    for candidate in PRECEDENCE_ORDER:
        if candidate in classes:
            return candidate
    return cast(ReplyClass, "not_relevant")
