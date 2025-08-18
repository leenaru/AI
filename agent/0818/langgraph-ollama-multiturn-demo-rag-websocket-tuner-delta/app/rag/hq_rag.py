from __future__ import annotations
import os, json, heapq
from typing import List, Tuple
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from app.common_types import ChatState
from app.rag.embed import get_embedder

INDEX_DIR = os.getenv("HQ_INDEX", "/data/index/faiss_hq")

def multi_query_expand(question: str, model_tag: str, n: int = 3) -> list[str]:
    sys = SystemMessage("질문의 의미를 보존하면서 검색 적합성이 높은 한국어 쿼리 3개를 JSON 배열로만 출력하세요.")
    user = HumanMessage(f"질문: {question}")
    chat = ChatOllama(model=model_tag, temperature=0.1)
    resp = chat.invoke([sys, user]).content.strip()
    try:
        arr = json.loads(resp)
        if isinstance(arr, list):
            return [str(x) for x in arr][:n]
    except Exception:
        pass
    return [question]

def retrieve_fuse(state: ChatState):
    emb = get_embedder()
    vs = FAISS.load_local(INDEX_DIR, embeddings=emb, allow_dangerous_deserialization=True)
    q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    model_tag = state.get("model")
    queries = multi_query_expand(q, model_tag, n=3)
    # RRF-style fusion
    scores: dict[str, float] = {}
    docs_map = {}
    for qx in queries:
        docs = vs.similarity_search(qx, k=6)
        for rank, d in enumerate(docs):
            key = d.page_content[:200]  # naive key
            docs_map[key] = d
            scores[key] = scores.get(key, 0.0) + 1.0/(rank+1)
    # top fused
    top = heapq.nlargest(6, scores.items(), key=lambda x: x[1])
    candidates = [docs_map[k] for k,_ in top]
    return {"candidates": candidates, "messages":[AIMessage(f"[HQ-RAG] MQE+RRF candidates={len(candidates)}")]}

def llm_rerank(state: ChatState):
    cand = state.get("candidates") or []
    if not cand:
        return {"snippets":"", "messages":[AIMessage("[HQ-RAG] no candidates")]}
    model_tag = state.get("model")
    sys = SystemMessage("주어진 질문과 후보 문서들을 0~1 사이 점수로 재랭킹하세요. JSON 배열로만 출력. 예: [[idx,score],...]")
    q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    items = "\n\n".join([f"[{i}] {d.page_content.strip()}" for i,d in enumerate(cand)])
    user = HumanMessage(f"[질문]\n{q}\n\n[후보]\n{items}")
    chat = ChatOllama(model=model_tag, temperature=0.0)
    resp = chat.invoke([sys,user]).content.strip()
    order = []
    try:
        arr = json.loads(resp)
        if isinstance(arr, list):
            order = sorted([(int(i), float(s)) for i,s in arr if 0<=int(i)<len(cand)], key=lambda x: x[1], reverse=True)
    except Exception:
        order = list(reversed(list(enumerate([0.5]*len(cand)))))
    picked = [cand[i] for i,_ in order[:4]]
    snippets = "\n\n".join([f"[{i+1}] {d.page_content.strip()}" for i,d in enumerate(picked)])
    return {"snippets": snippets, "messages":[AIMessage(f"[HQ-RAG] reranked={len(picked)}")]}

def generate(state: ChatState):
    from langchain_community.chat_models import ChatOllama
    sys = SystemMessage("한국어 HQ-RAG 어시스턴트. 근거 위주로 정확하게 답하고, 출처/문맥이 부족하면 모른다고 말하세요.")
    chat = ChatOllama(model=state.get("model"), temperature=0.2)
    q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    ctx = state.get("snippets") or ""
    user = HumanMessage(f"[질문]\n{q}\n\n[선정 컨텍스트]\n{ctx}\n\n요약→상세 순으로 답하고, 부족하면 한계를 명시.")
    ai = chat.invoke([sys,user])
    return {"messages":[ai]}

def build():
    g = StateGraph(ChatState | dict)
    g.add_node("retrieve_fuse", retrieve_fuse)
    g.add_node("llm_rerank", llm_rerank)
    g.add_node("generate", generate)
    g.set_entry_point("retrieve_fuse")
    g.add_edge("retrieve_fuse","llm_rerank")
    g.add_edge("llm_rerank","generate")
    g.add_edge("generate", END)
    return g.compile(checkpointer=MemorySaver())
