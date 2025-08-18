from __future__ import annotations
import argparse, os, glob
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.rag.embed import get_embedder

def build_index(docs_dir: str, out_dir: str, chunk_size=600, chunk_overlap=120):
    files = sorted(glob.glob(os.path.join(docs_dir, "**/*.*"), recursive=True))
    if not files:
        raise SystemExit(f"No docs found in {docs_dir}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    for fp in files:
        try:
            docs.extend(splitter.split_documents(TextLoader(fp, autodetect_encoding=True).load()))
        except Exception as e:
            print(f"Skip {fp}: {e}")
    emb = get_embedder()
    vs = FAISS.from_documents(docs, embedding=emb)
    os.makedirs(out_dir, exist_ok=True)
    vs.save_local(out_dir)
    print(f"[FAISS] saved to {out_dir} with {len(docs)} chunks")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="/data/docs")
    p.add_argument("--out",  default="/data/index/faiss_basic")
    args = p.parse_args()
    build_index(args.docs, args.out)
