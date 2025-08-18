from __future__ import annotations
import os, json, heapq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from app.common_types import ChatState
from app.rag.embed import get_embedder

INDEX_DIR = os.getenv("HQ_INDEX", "/data/index/faiss_hq")

def multi_query(question: str, model_tag: str, n=3):
    sys = SystemMessage("질문 의미를 유지하면서 검색 적합성이 높은 한국어 쿼리 3개를 JSON 배열로만 출력.")
    chat = ChatOllama(model=model_tag, temperature=0.0)
    resp = chat.invoke([sys, HumanMessage(question)]).content.strip()
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
    queries = multi_query(q, model_tag, n=3)
    scores = {}; docs_map = {}
    for qx in queries:
        docs = vs.similarity_search(qx, k=6)
        for rank, d in enumerate(docs):
            key = d.page_content[:200]
            docs_map[key] = d
            scores[key] = scores.get(key, 0.0) + 1.0/(rank+1)
    top = heapq.nlargest(6, scores.items(), key=lambda x: x[1])
    cand = [docs_map[k] for k,_ in top]
    return {"candidates": cand, "messages":[AIMessage(f"[HQ-RAG] 후보 {len(cand)}개")]}

def rerank(state: ChatState):
    cand = state.get("candidates") or []
    if not cand:
        return {"snippets":"", "messages":[AIMessage("[HQ-RAG] 후보 없음")]}
    chat = ChatOllama(model=state.get("model"), temperature=0.0)
    sys = SystemMessage("후보 문서를 질문 적합도로 0~1 점수화해 JSON 배열 [[idx,score],...] 만 출력.")
    q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    listing = "\n\n".join([f"[{i}] {d.page_content.strip()}" for i,d in enumerate(cand)])
    resp = chat.invoke([sys, HumanMessage(f"{q}\n\n{listing}")]).content.strip()
    order = []
    try:
        arr = json.loads(resp)
        order = sorted([(int(i), float(s)) for i,s in arr if 0<=int(i)<len(cand)], key=lambda x: x[1], reverse=True)
    except Exception:
        order = list(reversed(list(enumerate([0.5]*len(cand)))))
    picked = [cand[i] for i,_ in order[:4]]
    snippets = "\n\n".join([f"[{i+1}] {d.page_content.strip()}" for i,d in enumerate(picked)])
    return {"snippets": snippets}

def answer(state: ChatState):
    chat = ChatOllama(model=state.get("model"), temperature=0.2)
    sys = SystemMessage("한국어 HQ-RAG. 컨텍스트 근거 위주, 부족하면 모른다고. 요약→상세.")
    q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    ctx = state.get("snippets") or ""
    user = HumanMessage(f"[Q]\n{q}\n\n[CTX]\n{ctx}")
    ai = chat.invoke([sys] + state["messages"] + [user])
    return {"messages":[ai]}

def build():
    g = StateGraph(ChatState | dict)
    g.add_node("retrieve_fuse", retrieve_fuse)
    g.add_node("rerank", rerank)
    g.add_node("answer", answer)
    g.set_entry_point("retrieve_fuse")
    g.add_edge("retrieve_fuse","rerank")
    g.add_edge("rerank","answer")
    g.add_edge("answer", END)
    return g.compile(checkpointer=MemorySaver())
