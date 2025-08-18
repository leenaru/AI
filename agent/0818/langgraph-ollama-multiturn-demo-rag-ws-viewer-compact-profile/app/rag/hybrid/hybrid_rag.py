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
HYBRID_CONF = "/data/index/hybrid_config.json"

def load_bm25():
    with open(BM25_JSON, "r", encoding="utf-8") as f:
        obj = json.load(f)
    texts = obj.get("texts", [])
    sources = obj.get("sources", [])
    tokenized = [t.split() for t in texts]
    return BM25Okapi(tokenized), texts, sources

def hybrid_search(query: str, k=6) -> List[Tuple[str, float, str]]:

    # Load tuner config
    conf = {"weight_vec": 0.5, "weight_bm25": 0.5, "fuse":"max", "rrf_k":60}
    try:
        if os.path.exists(HYBRID_CONF):
            conf.update(json.load(open(HYBRID_CONF, "r", encoding="utf-8")))
    except Exception:
        pass
    wv = float(conf.get("weight_vec",0.5)); wb = float(conf.get("weight_bm25",0.5))
    fuse = str(conf.get("fuse","max")).lower(); rrf_k = int(conf.get("rrf_k",60))

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
    # Fusion
    pool = {}
    if fuse == "rrf":
        # rank-based Reciprocal Rank Fusion
        ranks_vec = {t[:1800]: i for i, (t,s,src) in enumerate(v_items)}
        ranks_bm  = {t[:1800]: i for i, (t,s,src) in enumerate(b_items)}
        keys = set(list(ranks_vec.keys()) + list(ranks_bm.keys()))
        for key in keys:
            rv = ranks_vec.get(key); rb = ranks_bm.get(key)
            sc = 0.0
            if rv is not None: sc += wv * 1.0/(rrf_k + rv + 1)
            if rb is not None: sc += wb * 1.0/(rrf_k + rb + 1)
            pool[key] = sc
    elif fuse == "sum":
        for t, s, src in v_items:
            key = t[:1800]; pool[key] = pool.get(key, 0.0) + wv*s
        for t, s, src in b_items:
            key = t[:1800]; pool[key] = pool.get(key, 0.0) + wb*s
    else:  # max
        for t, s, src in v_items + b_items:
            key = t[:1800]; pool[key] = max(pool.get(key, 0.0), (wv if (t,s,src) in v_items else wb)*s)
    ranked = sorted(pool.items(), key=lambda x: x[1], reverse=True)[:k]
    out = []
    for t, s in ranked:
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
