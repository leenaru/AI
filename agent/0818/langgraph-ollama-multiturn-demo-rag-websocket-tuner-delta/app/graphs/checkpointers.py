from __future__ import annotations
import os, sqlite3
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
from app.common_types import ChatState
from app.utils.model_tags import DEFAULT_MODEL

def llm_node(state: ChatState):
    chat = ChatOllama(model=state.get("model") or DEFAULT_MODEL, temperature=0.2)
    ai = chat.invoke(state["messages"])
    return {"messages":[ai]}

def build():
    backend = os.getenv("CHKPT_BACKEND", "memory").lower()
    checkpointer = None
    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
    elif backend == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver
        db = os.getenv("CHECKPOINT_DB", "/data/checkpoints.sqlite")
        os.makedirs(os.path.dirname(db), exist_ok=True)
        import sqlite3
        conn = sqlite3.connect(db, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    elif backend == "redis":
        # pip install redis
        from langgraph.checkpoint.redis import RedisSaver
        import redis
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = redis.Redis.from_url(url)
        checkpointer = RedisSaver(client, namespace=os.getenv("REDIS_NAMESPACE","lg"))
    elif backend == "postgres":
        # pip install psycopg2-binary
        from langgraph.checkpoint.postgres import PostgresSaver
        import psycopg2
        url = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/langgraph")
        conn = psycopg2.connect(url)
        checkpointer = PostgresSaver(conn)
    else:
        raise RuntimeError(f"Unknown CHKPT_BACKEND: {backend}")

    g = StateGraph(ChatState)
    g.add_node("chat", llm_node)
    g.set_entry_point("chat")
    g.add_edge("chat", END)
    return g.compile(checkpointer=checkpointer)
