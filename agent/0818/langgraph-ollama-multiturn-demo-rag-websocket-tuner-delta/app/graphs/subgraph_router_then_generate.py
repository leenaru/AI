from __future__ import annotations
import json, re
from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from app.common_types import ChatState
from app.utils.model_tags import ROUTER_MODEL, GENERATOR_MODEL

# 라우터 서브그래프 (LLM 판단)
def decide_next(state: ChatState):
    s = state.get("slots") or {}
    sys = SystemMessage("JSON으로만 출력하세요. action ∈ {ask_city,ask_days,plan}")
    user = HumanMessage(f"slots={s} → 다음 액션 추천")
    chat = ChatOllama(model=ROUTER_MODEL, temperature=0.0)
    resp = chat.invoke([sys]+state["messages"]+[user]).content.strip()
    try:
        action = json.loads(resp).get("action")
    except Exception:
        m=re.search(r'"action"\s*:\s*"([^"]+)"', resp)
        action = m.group(1) if m else "ask_city"
    if action not in {"ask_city","ask_days","plan"}:
        action="ask_city"
    return {"next_action": action, "messages":[AIMessage(f"[router] action={action}")]}

def ask_city(state: ChatState): return {"messages":[AIMessage("도시는 어디인가요?")], "pending":"city"}
def ask_days(state: ChatState): return {"messages":[AIMessage("며칠 계획인가요? (예: 3일)")], "pending":"days"}

def route_label(state: ChatState) -> Literal["city","days","plan"]:
    a = state.get("next_action") or "ask_city"
    return "city" if a=="ask_city" else ("days" if a=="ask_days" else "plan")

def build_router_subgraph():
    g = StateGraph(ChatState)
    g.add_node("decide", decide_next)
    g.add_node("ask_city", ask_city)
    g.add_node("ask_days", ask_days)
    g.set_entry_point("decide")
    g.add_conditional_edges("decide", route_label, {"city":"ask_city","days":"ask_days","plan":END})
    g.add_edge("ask_city", END)
    g.add_edge("ask_days", END)
    return g.compile()

# 생성 서브그래프
def generate_plan(state: ChatState):
    chat = ChatOllama(model=GENERATOR_MODEL, temperature=0.3)
    ai = chat.invoke([HumanMessage(f"slots={state.get('slots')}로 가족 친화 일정")])
    return {"messages":[ai], "done":True}

def build():
    g = StateGraph(ChatState)
    g.add_node("router", build_router_subgraph())
    g.add_node("generate", generate_plan)
    g.set_entry_point("router")
    g.add_edge("router", "generate")
    g.add_edge("generate", END)
    return g.compile()
