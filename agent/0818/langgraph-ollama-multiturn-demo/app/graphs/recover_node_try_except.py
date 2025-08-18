from __future__ import annotations
import random
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from app.common_types import ChatState
from app.utils.model_tags import DEFAULT_MODEL, GENERATOR_MODEL

def risky_tool(state: ChatState):
    if any("fail" in getattr(m,"content","").lower() for m in state["messages"]):
        raise RuntimeError("외부 API 오류(유도)")
    if random.random() < 0.2:
        raise RuntimeError("일시 장애(랜덤)")
    return {"messages":[AIMessage("도구 호출 성공: 결과 요약...")]}

def node_with_recovery(state: ChatState):
    try:
        return risky_tool(state)
    except Exception as e:
        return {"messages":[AIMessage(f"[오류] {e}. 대안 경로로 전환합니다.")], "error": True}

def recover(state: ChatState):
    chat = ChatOllama(model=GENERATOR_MODEL, temperature=0.1)
    ai = chat.invoke(state["messages"]+[HumanMessage("도구 불가. 최대한 유용한 대안을 제시해줘.")])
    return {"messages":[ai], "error": False}

def route(state: ChatState) -> Literal["ok","recover"]:
    return "recover" if state.get("error") else "ok"

def build():
    g = StateGraph(ChatState)
    g.add_node("step", node_with_recovery)
    g.add_node("recover", recover)
    g.set_entry_point("step")
    g.add_conditional_edges("step", route, {"ok": END, "recover": "recover"})
    g.add_edge("recover", END)
    return g.compile(checkpointer=MemorySaver())
