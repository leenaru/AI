from __future__ import annotations
import os, json, math
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from app.rag.embed import get_embedder

FAISS_DIR = os.getenv("HYBRID_FAISS", "/data/index/faiss_hybrid")
BM25_JSON = os.getenv("HYBRID_BM25", "/data/index/bm25.json")

def load_bm25():
    with open(BM25_JSON, "r", encoding="utf-8") as f:
        obj = json.load(f)
    texts = obj.get("texts", [])
    sources = obj.get("sources", [])
    tokenized = [t.split() for t in texts]
    return BM25Okapi(tokenized), texts, sources

def hybrid_search(query: str, k=6) -> List[Tuple[str, float, str]]:
    # Vector part
    vs = FAISS.load_local(FAISS_DIR, embeddings=get_embedder(), allow_dangerous_deserialization=True)
    vec_docs = vs.similarity_search_with_score(query, k=k)
    # Normalize vector scores (lower distance → higher score). Here, invert distance with soft transform.
    if vec_docs and isinstance(vec_docs[0][1], (float, int)):
        maxd = max(d for _, d in vec_docs)
        mind = min(d for _, d in vec_docs)
        rng = max(1e-6, (maxd - mind))
        v_items = [(doc.page_content, 1.0 - ((dist - mind) / rng), doc.metadata.get("source","")) for doc, dist in vec_docs]
    else:
        v_items = [(doc.page_content, 0.5, doc.metadata.get("source","")) for doc, _ in vec_docs]
    # BM25 part
    bm25, texts, sources = load_bm25()
    doc_scores = bm25.get_scores(query.split())
    # Top-k BM25
    bm_idx = sorted(range(len(texts)), key=lambda i: doc_scores[i], reverse=True)[:k]
    b_items = [(texts[i], float(doc_scores[i]), sources[i]) for i in bm_idx]
    # Normalize BM scores
    if b_items:
        maxs = max(s for _,s,_ in b_items); mins = min(s for _,s,_ in b_items)
        rng = max(1e-6, (maxs - mins))
        b_items = [(t, (s - mins)/rng if rng>0 else 0.5, src) for t,s,src in b_items]
    # Fuse by weighted sum
    pool = {}
    for t, s, src in v_items + b_items:
        key = t[:1800]
        pool[key] = max(pool.get(key, 0.0), s)  # max-pool
    ranked = sorted(pool.items(), key=lambda x: x[1], reverse=True)[:k]
    # Return (text, score, source)
    src_map = {t[:1800]: s for t,s,_ in v_items + b_items}
    out = []
    for t, s in ranked:
        # source best-effort
        src = ""
        for tx, sc, so in v_items + b_items:
            if tx[:1800]==t:
                src = so; break
        out.append((t, float(s), src))
    return out

def generate_answer(question: str, ctx_snippets: list[str], model_tag: str) -> str:
    sys = SystemMessage("한국어 하이브리드 RAG 어시스턴트. 컨텍스트만 근거로 사용하고, 부족하면 모른다고 답하세요.")
    user = HumanMessage(f"[질문]\n{question}\n\n[컨텍스트]\n" + "\n\n".join(ctx_snippets))
    chat = ChatOllama(model=model_tag, temperature=0.2)
    return chat.invoke([sys, user]).content
