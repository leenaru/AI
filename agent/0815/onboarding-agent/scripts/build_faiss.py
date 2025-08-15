import argparse, glob
from server.rag.pipeline import run_pipeline

def load_docs(path_glob: str):
    docs = []
    for p in glob.glob(path_glob):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            docs.append((p, f.read()))
    return docs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs/*.txt")
    args = ap.parse_args()
    docs = load_docs(args.docs)
    run_pipeline(docs)
    print("FAISS index built.")
