from __future__ import annotations
from typing import Annotated, TypedDict, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

AllowedAction = Literal["ask_city", "ask_days", "ask_kid", "ask_budget", "plan"]

class Slots(TypedDict, total=False):
    city: str
    days: int
    with_kid: bool
    budget: Literal["low", "mid", "high"]

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    slots: Slots
    pending: str | None
    model: str
    next_action: AllowedAction | None
    reason: str | None
    done: bool
    error: bool | None
    retry: int | None
