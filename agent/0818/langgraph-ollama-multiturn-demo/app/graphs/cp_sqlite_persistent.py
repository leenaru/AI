from __future__ import annotations
import sqlite3, os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.chat_models import ChatOllama
from app.common_types import ChatState
from app.utils.model_tags import DEFAULT_MODEL

DB_PATH = os.getenv("CHECKPOINT_DB", "/data/checkpoints.sqlite")

def llm_node(state: ChatState):
    chat = ChatOllama(model=state.get("model") or DEFAULT_MODEL, temperature=0.2)
    ai = chat.invoke(state["messages"])
    return {"messages":[ai]}

def build():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    saver = SqliteSaver(conn)
    sg = StateGraph(ChatState)
    sg.add_node("chat", llm_node)
    sg.set_entry_point("chat")
    sg.add_edge("chat", END)
    return sg.compile(checkpointer=saver)
