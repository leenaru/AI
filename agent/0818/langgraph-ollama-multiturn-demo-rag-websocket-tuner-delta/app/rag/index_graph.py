from __future__ import annotations
import argparse, os, glob, json, re
import networkx as nx
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from app.utils.model_tags import GENERATOR_MODEL

PROMPT = ("""다음 텍스트에서 '엔티티(명사구)'와 그들 사이의 '관계(동사/술어)'를 추출하세요.
출력은 JSON만, 예: {"entities":["타롱가 동물원","시드니"], "relations":[["타롱가 동물원","위치","시드니"]]}.
텍스트:
""")

def extract_entities_relations(text: str) -> dict:
    chat = ChatOllama(model=GENERATOR_MODEL, temperature=0.0)
    from langchain_core.messages import HumanMessage
    resp = chat.invoke([HumanMessage(PROMPT + text[:1500])]).content.strip()
    try:
        data = json.loads(resp)
        ents = list(dict.fromkeys(data.get("entities", [])))
        rels = [tuple(r) for r in data.get("relations", []) if isinstance(r, (list, tuple)) and len(r)==3]
        return {"entities": ents, "relations": rels}
    except Exception:
        # 매우 단순한 백업: 고윳명사/명사 추정(한글/영문 대문자 시작 단어들)
        ents = list(set(re.findall(r"[가-힣A-Z][가-힣A-Za-z0-9_ ]{1,20}", text)))
        return {"entities": ents, "relations": []}

def build_graph(docs_dir: str, out_path: str, chunk_size=800, chunk_overlap=100):
    files = sorted(glob.glob(os.path.join(docs_dir, "**/*.*"), recursive=True))
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    G = nx.MultiDiGraph()
    for fp in files:
        try:
            raw = TextLoader(fp, autodetect_encoding=True).load()[0]
        except Exception as e:
            print(f"Skip {fp}: {e}")
            continue
        for chunk in splitter.split_documents([raw]):
            info = extract_entities_relations(chunk.page_content)
            for e in info["entities"]:
                G.add_node(e, type="entity")
            for h, rel, t in info["relations"]:
                G.add_edge(h, t, rel=rel, source=fp)
    data = nx.readwrite.json_graph.node_link_data(G)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[GraphRAG] nodes={G.number_of_nodes()} edges={G.number_of_edges()} saved to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="/data/docs")
    p.add_argument("--out",  default="/data/index/graph.json")
    args = p.parse_args()
    build_graph(args.docs, args.out)
