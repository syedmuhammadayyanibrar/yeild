from dataclasses import dataclass
from typing import Any


@dataclass
class QualityGate:
    id: str
    description: str
    value: Any
    threshold: Any
    operator: str
    status: str

def evaluate_gate(
    gate_id: str,
    description: str,
    val: float | None,
    threshold: Any,
    operator: str,
    skip_reason: str | None = None,
) -> QualityGate:
    if val is None or skip_reason is not None:
        return QualityGate(
            id=gate_id,
            description=description,
            value=f"SKIPPED ({skip_reason or 'No API key provided'})",
            threshold=threshold,
            operator=operator,
            status="SKIPPED",
        )

    passed = False
    v = float(val)
    if operator == ">=":
        passed = v >= float(threshold)
    elif operator == "<=":
        passed = v <= float(threshold)
    elif operator == "==":
        passed = v == float(threshold)
    elif operator == "in":
        lo = float(threshold[0])
        hi = float(threshold[1])
        passed = lo <= v <= hi
    elif operator == "<":
        passed = v < float(threshold)

    return QualityGate(
        id=gate_id,
        description=description,
        value=round(val, 4) if isinstance(val, float) else val,
        threshold=threshold,
        operator=operator,
        status="PASS" if passed else "FAIL",
    )
