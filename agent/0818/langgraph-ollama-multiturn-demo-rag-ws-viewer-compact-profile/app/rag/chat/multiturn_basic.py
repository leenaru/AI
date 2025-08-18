from __future__ import annotations
import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from app.common_types import ChatState
from app.rag.embed import get_embedder

INDEX_DIR = os.getenv("BASIC_INDEX", "/data/index/faiss_basic")

def retrieve(state: ChatState):
    emb = get_embedder()
    vs = FAISS.load_local(INDEX_DIR, embeddings=emb, allow_dangerous_deserialization=True)
    # 마지막 사용자 메시지를 질의로 사용
    q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    docs = vs.similarity_search(q, k=5)
    snippets = "\n\n".join([f"[{i+1}] {d.page_content.strip()}" for i, d in enumerate(docs)])
    return {"snippets": snippets, "messages":[AIMessage(content=f"[RAG] {len(docs)}개 문맥 로드")]}

def answer(state: ChatState):
    sys = SystemMessage("한국어 RAG 어시스턴트. 주어진 컨텍스트를 우선 근거로 사용하고, 부족하면 '모른다'고 답하세요. 요약→상세 순.")
    chat = ChatOllama(model=state.get("model"), temperature=0.2)
    q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    ctx = state.get("snippets") or ""
    user = HumanMessage(f"[질문]\n{q}\n\n[컨텍스트]\n{ctx}")
    ai = chat.invoke([sys] + state["messages"] + [user])
    return {"messages":[ai]}

def build():
    g = StateGraph(ChatState | dict)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile(checkpointer=MemorySaver())
