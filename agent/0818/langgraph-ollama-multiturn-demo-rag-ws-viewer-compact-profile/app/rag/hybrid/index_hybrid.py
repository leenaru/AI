from __future__ import annotations
import argparse, os, glob, json, re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from app.rag.embed import get_embedder
import hashlib, json

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

FAISS_IDS = "/data/index/faiss_ids.json"

def build_faiss(docs_dir: str, out_dir: str, chunk_size=600, chunk_overlap=120):
    files = sorted(glob.glob(os.path.join(docs_dir, "**/*.*"), recursive=True))
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    for fp in files:
        try:
            docs.extend(splitter.split_documents(TextLoader(fp, autodetect_encoding=True).load()))
        except Exception as e:
            print("Skip", fp, e)
    emb = get_embedder()
    texts = [d.page_content for d in docs]
metas = [d.metadata for d in docs]
# ids: source::sha1(chunk)
ids = []
for d in docs:
    src = d.metadata.get('source','unknown')
    cid = hashlib.sha1((src+'|'+d.page_content).encode('utf-8')).hexdigest()[:16]
    ids.append(f"{src}::{cid}")
vs = FAISS.from_texts(texts, embedding=emb, metadatas=metas, ids=ids)
os.makedirs(out_dir, exist_ok=True)
vs.save_local(out_dir)
# write ids map
m = {}
for i,d in enumerate(docs):
    src = d.metadata.get('source','unknown')
    m.setdefault(src, []).append(ids[i])
os.makedirs(os.path.dirname(FAISS_IDS), exist_ok=True)
json.dump(m, open(FAISS_IDS,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
print(f"[Hybrid] FAISS saved: {out_dir} (chunks={len(docs)}) and ids map -> {FAISS_IDS}")

def build_bm25(docs_dir: str, out_json: str):
    files = sorted(glob.glob(os.path.join(docs_dir, "**/*.*"), recursive=True))
    texts = []
    srcs = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
        except Exception as e:
            print("Skip", fp, e); continue
        texts.append(normalize_ws(txt))
        srcs.append(fp)
    # 토큰은 런타임에서 생성하므로 텍스트/소스만 저장
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "sources": srcs}, f, ensure_ascii=False)
    print(f"[Hybrid] BM25 json saved: {out_json} (docs={len(texts)})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="/data/docs")
    p.add_argument("--faiss_out", default="/data/index/faiss_hybrid")
    p.add_argument("--bm25_out", default="/data/index/bm25.json")
    args = p.parse_args()
    build_faiss(args.docs, args.faiss_out)
    build_bm25(args.docs, args.bm25_out)
