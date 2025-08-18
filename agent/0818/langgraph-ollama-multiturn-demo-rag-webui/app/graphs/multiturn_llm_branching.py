from __future__ import annotations
import json, re
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from app.common_types import ChatState, Slots
from app.utils.model_tags import ROUTER_MODEL, GENERATOR_MODEL, DEFAULT_MODEL

ALLOWED = ["ask_city","ask_days","ask_kid","ask_budget","plan"]

def parse_days(text: str):
    import re
    m=re.search(r"(\d+)\s*일", text)
    return int(m.group(1)) if m else None

def parse_with_kid(text: str):
    kor=text
    pos=["아이","아기","자녀","유아","초등","5살","6살","7살"]
    neg=["아이 없음","아이없","혼자","어른만","성인만"]
    if any(k in kor for k in pos): return True
    if any(k in kor for k in neg): return False
    if any(k in kor for k in ["예","네","있"]): return True
    if any(k in kor for k in ["아니","없"]): return False
    return None

def parse_budget(text: str):
    kor=text
    if any(k in kor for k in ["저렴","가성비","저가","절약","값싼"]): return "low"
    if any(k in kor for k in ["중간","보통","무난","중간정도"]): return "mid"
    if any(k in kor for k in ["고급","프리미엄","럭셔리","호화"]): return "high"
    return None

def ingest(state: ChatState):
    pending = state.get("pending")
    if not pending:
        return {}
    last = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last:
        return {}
    text = last.content.strip()
    s = dict(state.get("slots") or {})
    if pending=="city": s["city"]=text
    elif pending=="days":
        d=parse_days(text)
        if d: s["days"]=d
    elif pending=="with_kid":
        w=parse_with_kid(text)
        if w is not None: s["with_kid"]=w
    elif pending=="budget":
        b=parse_budget(text)
        if b: s["budget"]=b
    return {"slots":s, "pending":None}

ROUTER_SYSTEM = SystemMessage(content=(
    "당신은 대화 오케스트레이터입니다. 다음 액션을 정확히 하나 선택하여 JSON만 출력하세요."
))

ROUTER_USER_TMPL = """
[현재 슬롯]
city={city}, days={days}, with_kid={with_kid}, budget={budget}
허용 액션: ask_city, ask_days, ask_kid, ask_budget, plan
JSON 예: {{"action":"ask_days","reason":"일수 미확정"}}
"""

def _coerce_allowed(action, slots: Slots):
    if action in ALLOWED:
        return action
    if not slots.get("city"): return "ask_city"
    if not slots.get("days"): return "ask_days"
    if "with_kid" not in slots: return "ask_kid"
    if not slots.get("budget"): return "ask_budget"
    return "plan"

def decide_next(state: ChatState):
    s = state.get("slots") or {}
    chat = ChatOllama(model=ROUTER_MODEL, temperature=0.0)
    user = HumanMessage(content=ROUTER_USER_TMPL.format(**{
        "city": s.get("city"),
        "days": s.get("days"),
        "with_kid": s.get("with_kid"),
        "budget": s.get("budget"),
    }))
    resp = chat.invoke([ROUTER_SYSTEM] + state["messages"] + [user])
    raw = resp.content.strip()
    action=None; reason=None
    try:
        data=json.loads(raw); action=data.get("action"); reason=data.get("reason")
    except Exception:
        m=re.search(r'"action"\s*:\s*"([^"]+)"', raw); action=m.group(1) if m else None
        m2=re.search(r'"reason"\s*:\s*"([^"]+)"', raw); reason=m2.group(1) if m2 else None
    action=_coerce_allowed(action, s)
    if not reason: reason=f"LLM route → {action}"
    obs=AIMessage(content=f"[router] next_action={action}, reason={reason}")
    return {"messages":[obs], "next_action":action, "reason":reason}

def ask_city(state: ChatState): return {"messages":[AIMessage("도시는 어디인가요?")], "pending":"city"}
def ask_days(state: ChatState): return {"messages":[AIMessage("여행은 며칠인가요? (예: 3일)")], "pending":"days"}
def ask_kid(state: ChatState): return {"messages":[AIMessage("아이 동반 여부/나이를 알려주세요.")], "pending":"with_kid"}
def ask_budget(state: ChatState): return {"messages":[AIMessage("예산 수준(저렴/중간/고급) 중 선택해주세요.")], "pending":"budget"}

def plan(state: ChatState):
    s=state.get("slots") or {}
    sys = SystemMessage("한국어 여행 플래너: 요약→상세, 공손체")
    chat = ChatOllama(model=GENERATOR_MODEL, temperature=0.3)
    user = HumanMessage(f"slots={s} 기준 가족 친화 일정. 맛집/동선/체크리스트 포함")
    ai = chat.invoke([sys]+state["messages"]+[user])
    return {"messages":[ai], "done":True}

def _read_next(state: ChatState) -> str:
    return state.get("next_action") or "ask_city"

def build():
    g = StateGraph(ChatState)
    g.add_node("ingest", ingest)
    g.add_node("decide_next", decide_next)
    for n in ["ask_city","ask_days","ask_kid","ask_budget","plan"]:
        g.add_node(n, globals()[n])
    g.set_entry_point("ingest")
    g.add_edge("ingest","decide_next")
    g.add_conditional_edges("decide_next", _read_next, {
        "ask_city":"ask_city","ask_days":"ask_days","ask_kid":"ask_kid","ask_budget":"ask_budget","plan":"plan"
    })
    for n in ["ask_city","ask_days","ask_kid","ask_budget","plan"]:
        g.add_edge(n, END)
    return g.compile(checkpointer=MemorySaver())
