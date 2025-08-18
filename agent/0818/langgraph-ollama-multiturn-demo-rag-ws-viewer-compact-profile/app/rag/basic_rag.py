from __future__ import annotations
import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.common_types import ChatState
from app.rag.embed import get_embedder

INDEX_DIR = os.getenv("BASIC_INDEX", "/data/index/faiss_basic")

def retrieve(state: ChatState):
    emb = get_embedder()
    vs = FAISS.load_local(INDEX_DIR, embeddings=emb, allow_dangerous_deserialization=True)
    q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    docs = vs.similarity_search(q, k=4)
    snippets = "\n\n".join([f"[{i+1}] {d.page_content.strip()}" for i, d in enumerate(docs)])
    return {"messages":[AIMessage(content=f"[RAG] retrieved {len(docs)} chunks")], "snippets": snippets}

def generate(state: ChatState):
    sys = SystemMessage("당신은 엄격한 한국어 RAG 어시스턴트입니다. 주어진 컨텍스트만 사용하고, 모르면 모른다고 답하세요.")
    chat = ChatOllama(model=state.get("model"), temperature=0.2)
    q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    ctx = state.get("snippets") or ""
    user = HumanMessage(f"[질문]\n{q}\n\n[컨텍스트]\n{ctx}\n\n컨텍스트를 근거로 간결한 요약→상세 순으로 답하세요.")
    ai = chat.invoke([sys,user])
    return {"messages":[ai]}

def build():
    g = StateGraph(ChatState | dict)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve","generate")
    g.add_edge("generate", END)
    return g.compile(checkpointer=MemorySaver())
