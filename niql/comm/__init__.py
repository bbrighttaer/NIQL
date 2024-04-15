import dataclasses
from enum import auto
from strenum import StrEnum
from typing import Any

from .in_process import InterAgentComm


class CommMsg(StrEnum):
    OBSERVATION = auto()


@dataclasses.dataclass
class Message:
    msg_type: CommMsg
    msg: Any
