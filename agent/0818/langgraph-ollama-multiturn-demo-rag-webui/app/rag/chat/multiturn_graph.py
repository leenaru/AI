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

def seed(state: ChatState):
    chat = ChatOllama(model=GENERATOR_MODEL, temperature=0.0)
    q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    resp = chat.invoke([HumanMessage('질문 핵심 엔티티 3개 이내를 JSON 배열로만 출력.' + "\n질문: " + q)]).content.strip()
    try:
        arr = json.loads(resp); seeds = [str(x) for x in arr][:3]
    except Exception:
        seeds = [q[:20]]
    return {"seeds": seeds, "messages":[AIMessage(f"[GraphRAG] seeds={seeds}") ]}

def hop(state: ChatState, max_hops=2):
    G = load_graph()
    seeds = state.get("seeds") or []
    edges = []
    for s in seeds:
        if s not in G: continue
        frontier=[(s,0)]; visited=set()
        while frontier:
            n,h=frontier.pop(0)
            if (n,h) in visited or h>max_hops: continue
            visited.add((n,h))
            for _,t,data in G.out_edges(n, data=True):
                edges.append((n,t,data.get("rel","")))
                frontier.append((t,h+1))
    ctx = "\n".join([f"{h}-[{r}]->{t}" for h,t,r in edges[:50]])
    return {"graph_ctx": ctx}

def answer(state: ChatState):
    chat = ChatOllama(model=GENERATOR_MODEL, temperature=0.2)
    sys = SystemMessage("그래프 경로를 근거로 한국어로 답하세요. 모르면 모른다고.")
    q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    ctx = state.get("graph_ctx") or ""
    ai = chat.invoke([sys] + state["messages"] + [HumanMessage(f"[Q]\n{q}\n\n[GRAPH]\n{ctx}")])
    return {"messages":[ai]}

def build():
    g = StateGraph(ChatState | dict)
    g.add_node("seed", seed)
    g.add_node("hop", hop)
    g.add_node("answer", answer)
    g.set_entry_point("seed")
    g.add_edge("seed","hop")
    g.add_edge("hop","answer")
    g.add_edge("answer", END)
    return g.compile(checkpointer=MemorySaver())
