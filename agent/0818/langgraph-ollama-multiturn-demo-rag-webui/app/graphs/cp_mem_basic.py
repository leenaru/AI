from __future__ import annotations
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
from app.common_types import ChatState
from app.utils.model_tags import DEFAULT_MODEL

def llm_node(state: ChatState):
    chat = ChatOllama(model=state.get("model") or DEFAULT_MODEL, temperature=0.3)
    ai = chat.invoke(state["messages"])
    return {"messages":[ai]}

def build():
    sg = StateGraph(ChatState)
    sg.add_node("chat", llm_node)
    sg.set_entry_point("chat")
    sg.add_edge("chat", END)
    return sg.compile(checkpointer=MemorySaver())
