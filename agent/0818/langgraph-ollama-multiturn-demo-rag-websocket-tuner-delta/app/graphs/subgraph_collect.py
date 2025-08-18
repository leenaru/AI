from __future__ import annotations
import re
from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from app.common_types import ChatState

def ask_city(state: ChatState):   return {"messages":[AIMessage("어느 도시로 가시나요?")], "pending":"city"}
def ask_days(state: ChatState):   return {"messages":[AIMessage("며칠 일정인가요? (예: 3일)")], "pending":"days"}
def collect_route(state: ChatState) -> Literal["city","days","done"]:
    s = state.get("slots") or {}
    if not s.get("city"): return "city"
    if not s.get("days"): return "days"
    return "done"

def ingest(state: ChatState):
    last = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last or not state.get("pending"): return {}
    s = dict(state.get("slots") or {})
    if state["pending"]=="city": s["city"]=last.content.strip()
    if state["pending"]=="days":
        m=re.search(r"(\d+)\s*일",last.content)
        if m: s["days"]=int(m.group(1))
    return {"slots":s, "pending":None}

def build_collect_subgraph():
    g = StateGraph(ChatState)
    g.add_node("ingest", ingest)
    g.add_node("ask_city", ask_city)
    g.add_node("ask_days", ask_days)
    g.set_entry_point("ingest")
    g.add_conditional_edges("ingest", collect_route,
        {"city":"ask_city","days":"ask_days","done":END})
    g.add_edge("ask_city", END)
    g.add_edge("ask_days", END)
    return g.compile()

# 상위 그래프
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from app.utils.model_tags import DEFAULT_MODEL

def plan(state: ChatState):
    sys = SystemMessage("한국어로 공손히, 요약→상세 순")
    chat = ChatOllama(model=state.get("model") or DEFAULT_MODEL)
    ai = chat.invoke([sys]+state["messages"]+[HumanMessage(f"slots={state.get('slots')} 기반 일정 제안")])
    return {"messages":[ai], "done":True}

def build():
    sg = StateGraph(ChatState)
    sg.add_node("collect", build_collect_subgraph())
    sg.add_node("plan", plan)
    sg.set_entry_point("collect")
    sg.add_edge("collect","plan")
    sg.add_edge("plan", END)
    return sg.compile()
