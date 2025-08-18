from __future__ import annotations
import time
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from app.common_types import ChatState
from app.utils.model_tags import DEFAULT_MODEL, GENERATOR_MODEL

MAX_RETRY = 2

def gen_with_retry(state: ChatState):
    retry = int(state.get("retry") or 0)
    model = state.get("model") or DEFAULT_MODEL
    try:
        chat = ChatOllama(model=model, temperature=0.2)
        ai = chat.invoke(state["messages"])
        return {"messages":[ai], "error": False, "retry": retry}
    except Exception as e:
        if retry < MAX_RETRY:
            time.sleep(0.5 * (retry+1))
            return {"messages":[AIMessage(f"[경고] 모델 {model} 실패, 재시도={retry+1}")],
                    "error": True, "retry": retry+1}
        fallback = GENERATOR_MODEL  # 폴백(일반적으로 더 안정적/다른 계열)
        chat = ChatOllama(model=fallback, temperature=0.2)
        ai = chat.invoke(state["messages"]+[HumanMessage(f"메인 모델 실패. 폴백({fallback})로 처리")])
        return {"messages":[ai], "error": False, "retry": retry}

def route(state: ChatState) -> Literal["retry","done"]:
    return "retry" if state.get("error") else "done"

def build():
    g = StateGraph(ChatState)
    g.add_node("gen", gen_with_retry)
    g.set_entry_point("gen")
    g.add_conditional_edges("gen", route, {"retry":"gen","done":END})
    return g.compile(checkpointer=MemorySaver())
