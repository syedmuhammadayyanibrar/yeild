from typing import NamedTuple

from loop.schema import (
    ALL_ARMS,
    OpenerType,
    SignalType,
)


class ArmDef(NamedTuple):
    arm_id: str
    signal_type: SignalType
    opener_type: OpenerType

def get_arm_definitions() -> list[ArmDef]:
    defs: list[ArmDef] = []
    for arm in ALL_ARMS:
        s_str, o_str = arm.split(":")
        defs.append(ArmDef(arm_id=arm, signal_type=s_str, opener_type=o_str))
    return defs

def get_arm_index(arm_id: str) -> int:
    return ALL_ARMS.index(arm_id)

def get_arm_by_index(index: int) -> str:
    return ALL_ARMS[index]
