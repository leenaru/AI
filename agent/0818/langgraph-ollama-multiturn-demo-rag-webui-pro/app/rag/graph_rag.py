from __future__ import annotations
import os, json, networkx as nx
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from app.common_types import ChatState
from app.utils.model_tags import GENERATOR_MODEL

GRAPH_PATH = os.getenv("GRAPH_INDEX", "/data/index/graph.json")

def load_graph():
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.readwrite.json_graph.node_link_graph(data, directed=True, multigraph=True)

def seed_entities(state: ChatState):
    chat = ChatOllama(model=GENERATOR_MODEL, temperature=0.0)
    q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    from langchain_core.messages import HumanMessage
    prompt = '질문에서 핵심 엔티티(명사구) 3개 이내를 JSON 배열로만 출력하세요.'
    resp = chat.invoke([HumanMessage(prompt + "\n질문: " + q)]).content.strip()
    try:
        arr = json.loads(resp)
        seeds = [str(x) for x in arr][:3]
    except Exception:
        seeds = [q[:20]]
    return {"seeds": seeds, "messages":[AIMessage(f"[GraphRAG] seeds={seeds}")]}

def hop_collect(state: ChatState, max_hops=2):
    G = load_graph()
    seeds = state.get("seeds") or []
    visited = set()
    edges = []
    for s in seeds:
        if s not in G:
            continue
        frontier = [(s, 0)]
        while frontier:
            node, hop = frontier.pop(0)
            if (node, hop) in visited or hop>max_hops: 
                continue
            visited.add((node, hop))
            for _, t, data in G.out_edges(node, data=True):
                edges.append((node, t, data.get("rel","")))
                frontier.append((t, hop+1))
    # 간단 텍스트 컨텍스트로 변환
    lines = [f"{h} -[{r}]-> {t}" for h,t,r in edges[:50]]
    ctx = "\n".join(lines)
    return {"graph_ctx": ctx, "messages":[AIMessage(f"[GraphRAG] hops={len(edges)}") ]}

def answer(state: ChatState):
    chat = ChatOllama(model=GENERATOR_MODEL, temperature=0.2)
    q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    ctx = state.get("graph_ctx") or ""
    sys = SystemMessage("그래프 컨텍스트(엔티티-관계 경로)를 근거로 한국어로 답하세요. 모르면 모른다고 답하세요.")
    user = HumanMessage(f"[질문]\n{q}\n\n[그래프 경로]\n{ctx}")
    ai = chat.invoke([sys,user])
    return {"messages":[ai]}

def build():
    g = StateGraph(ChatState | dict)
    g.add_node("seed_entities", seed_entities)
    g.add_node("hop_collect", hop_collect)
    g.add_node("answer", answer)
    g.set_entry_point("seed_entities")
    g.add_edge("seed_entities","hop_collect")
    g.add_edge("hop_collect","answer")
    g.add_edge("answer", END)
    return g.compile(checkpointer=MemorySaver())
