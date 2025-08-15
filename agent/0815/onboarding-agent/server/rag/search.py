import os, numpy as np, faiss, pandas as pd, time
from typing import Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from .bm25 import BM25

PARQUET_DIR = "data/parquet"
FAISS_DIR = "data/faiss"
_emb = SentenceTransformer("intfloat/multilingual-e5-small")
_ce = None

cfg = {
  "k_dense": 40, "k_sparse": 40, "k_graph": 2,
  "weights": {"alpha":0.55,"beta":0.25,"gamma":0.15,"delta":0.05},
  "cross_encoder": {"enabled": True, "top_m": 10, "model": "BAAI/bge-reranker-v2-m3"}
}

def _load_units():
    return pd.read_parquet(os.path.join(PARQUET_DIR, "units.parquet"))

def _faiss_index():
    return faiss.read_index(os.path.join(FAISS_DIR, "hnsw.index"))

_units_cache = None
_bm25 = None

def hybrid_search(q: str) -> Dict[str, Any]:
    global _units_cache, _bm25, _ce
    if _units_cache is None:
        _units_cache = _load_units()
        _bm25 = BM25(_units_cache["summary"].tolist())
    # DENSE
    qv = _emb.encode(["query: "+q], normalize_embeddings=True).astype("float32")
    index = _faiss_index()
    D, I = index.search(qv, cfg["k_dense"])
    dense_scores = 1 - (D[0])

    # SPARSE
    idx_bm25, bm25_scores = _bm25.search(q, k=cfg["k_sparse"]) if cfg["k_sparse"]>0 else ([],[])

    # GRAPH proxy: recency
    now = int(time.time())
    recency = (now - _units_cache.loc[I[0], "created_at"]).clip(lower=1)
    recency_scores = 1/np.log(recency+2.71828)

    candidates = {}
    def add(idx, s_dense=0, s_bm25=0, s_graph=0, s_rec=0):
        if idx not in candidates:
            candidates[idx] = {"dense":0,"bm25":0,"graph":0,"recency":0}
        candidates[idx]["dense"] = max(candidates[idx]["dense"], s_dense)
        candidates[idx]["bm25"] = max(candidates[idx]["bm25"], s_bm25)
        candidates[idx]["graph"] = max(candidates[idx]["graph"], s_graph)
        candidates[idx]["recency"] = max(candidates[idx]["recency"], s_rec)

    for rank, idx in enumerate(I[0]):
        add(int(idx), s_dense=float(dense_scores[rank]), s_rec=float(recency_scores[rank]))
    for rank, idx in enumerate(idx_bm25):
        add(int(idx), s_bm25=float(bm25_scores[rank]))

    w = cfg["weights"]
    for k,v in candidates.items():
        v["score"] = w["alpha"]*v["dense"] + w["beta"]*v["bm25"] + w["gamma"]*v["graph"] + w["delta"]*v["recency"]

    top = sorted(candidates.items(), key=lambda kv: kv[1]["score"], reverse=True)[:cfg["cross_encoder"]["top_m"]]
    rows = [(i, _units_cache.iloc[i]["summary"]) for i,_ in top]

    if cfg["cross_encoder"]["enabled"]:
        if _ce is None:
            _ce = CrossEncoder(cfg["cross_encoder"]["model"])
        pairs = [(q, text) for _, text in rows]
        ces = _ce.predict(pairs)
        for idx, (i, _) in enumerate(rows):
            candidates[i]["score"] = float(ces[idx])
        top = sorted(candidates.items(), key=lambda kv: kv[1]["score"], reverse=True)[:10]

    passages = []
    for i,_ in top:
        row = _units_cache.iloc[i]
        passages.append({"id": str(row["unit_id"]), "text": row["summary"], "meta": {"doc_id": row["doc_id"]}, "score": candidates[i]["score"]})

    return {"passages": passages, "citations": [p["id"] for p in passages]}
