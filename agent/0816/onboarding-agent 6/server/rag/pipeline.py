import os, uuid, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, faiss
from sentence_transformers import SentenceTransformer
PARQUET_DIR, FAISS_DIR, EMB_DIM = "data/parquet", "data/faiss", 768
_model = SentenceTransformer("intfloat/multilingual-e5-small")
def graphify(docs):
  units, edges = [], []
  for doc_id, text in docs:
    chunks=[t.strip() for t in text.split("\n\n") if t.strip()]
    for c in chunks:
      uid=str(uuid.uuid4())
      units.append({"unit_id":uid,"doc_id":doc_id,"text":c,"created_at":int(time.time())})
      edges.append({"src":uid,"rel":"MENTIONS","tgt":doc_id})
  return pd.DataFrame(units), pd.DataFrame(edges)
def summarize_units(df): df["summary"]=df["text"].str.slice(0,200); return df
def build_faiss(df):
  os.makedirs(FAISS_DIR, exist_ok=True)
  texts=df["summary"].tolist()
  embs=_model.encode(["query: "+t for t in texts], normalize_embeddings=True)
  index=faiss.IndexHNSWFlat(EMB_DIM,32); index.hnsw.efConstruction=200
  index.add(np.array(embs).astype("float32"))
  faiss.write_index(index, os.path.join(FAISS_DIR,"hnsw.index"))
def save_parquet(df_units, df_edges):
  os.makedirs(PARQUET_DIR, exist_ok=True)
  pq.write_table(pa.Table.from_pandas(df_units), os.path.join(PARQUET_DIR,"units.parquet"))
  pq.write_table(pa.Table.from_pandas(df_edges), os.path.join(PARQUET_DIR,"edges.parquet"))
def run_pipeline(docs):
  units, edges = graphify(docs); units=summarize_units(units); save_parquet(units, edges); build_faiss(units)
