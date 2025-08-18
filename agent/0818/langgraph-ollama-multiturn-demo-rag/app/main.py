from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage
from app.utils.model_tags import DEFAULT_MODEL, ROUTER_MODEL, GENERATOR_MODEL, OLLAMA_HOST

# 그래프 빌더들
from app.graphs import cp_mem_basic, cp_sqlite_persistent, subgraph_collect, subgraph_router_then_generate, multiturn_llm_branching, recover_node_try_except, recover_retry_fallback
# RAG graphs
from app.rag import basic_rag as rag_basic
from app.rag import hq_rag as rag_hq
from app.rag import graph_rag as rag_graph
from pydantic import BaseModel


app = FastAPI(title="LangGraph + Ollama Examples")

def get_graph(name: str):
    if name == "basic":
        return cp_mem_basic.build()
    if name == "sqlite":
        return cp_sqlite_persistent.build()
    if name == "collect_plan":
        return subgraph_collect.build()
    if name == "router_generate":
        return subgraph_router_then_generate.build()
    if name == "llm_router_multiturn":
        return multiturn_llm_branching.build()
    if name == "recover_try":
        return recover_node_try_except.build()
    if name == "retry_fallback":
        return recover_retry_fallback.build()
    raise HTTPException(status_code=404, detail=f"unknown graph: {name}")

@app.get("/health")
def health():
    return {"status":"ok", "ollama_host": OLLAMA_HOST, "default_model": DEFAULT_MODEL, "router_model": ROUTER_MODEL, "generator_model": GENERATOR_MODEL}

class ChatReq(BaseModel):
    prompt: str
    graph: str = "llm_router_multiturn"
    thread_id: str = "default"
    model: str | None = None

class ChatResp(BaseModel):
    reply: str
    model: str

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    graph = get_graph(req.graph)
    cfg = {"configurable":{"thread_id": req.thread_id}}
    inputs = {"messages":[HumanMessage(req.prompt)], "model": req.model or DEFAULT_MODEL}
    res = graph.invoke(inputs, config=cfg)
    msg = res["messages"][-1].content if res.get("messages") else ""
    return ChatResp(reply=msg, model=req.model or DEFAULT_MODEL)

@app.get("/sse/values")
async def sse_values(prompt: str, graph: str = "collect_plan", thread_id: str = "sse-values", model: str | None = None):
    g = get_graph(graph)
    cfg = {"configurable":{"thread_id": thread_id}}
    inputs = {"messages":[HumanMessage(prompt)]}
    if model: inputs["model"]=model
    def gen():
        for chunk in g.stream(inputs, config=cfg, stream_mode="values"):
            # values 스트림: 상태 스냅샷
            keys = list(chunk.keys())
            yield {"event":"value", "data": str(keys)}
        yield {"event":"done", "data":"[DONE]"}
    return EventSourceResponse(gen())

@app.get("/sse/updates")
async def sse_updates(prompt: str, graph: str = "router_generate", thread_id: str = "sse-updates", model: str | None = None):
    g = get_graph(graph)
    cfg = {"configurable":{"thread_id": thread_id}}
    inputs = {"messages":[HumanMessage(prompt)]}
    if model: inputs["model"]=model
    def gen():
        for upd in g.stream(inputs, config=cfg, stream_mode="updates"):
            for node, payload in upd.items():
                yield {"event":"update", "data": f"{node}:{list(payload.keys())}"}
        yield {"event":"done", "data":"[DONE]"}
    return EventSourceResponse(gen())


class RagReq(BaseModel):
    query: str
    thread_id: str = "rag"
    model: str | None = None

@app.post("/rag/basic")
def rag_basic_endpoint(req: RagReq):
    g = rag_basic.build()
    cfg = {"configurable":{"thread_id": req.thread_id}}
    res = g.invoke({"messages":[HumanMessage(req.query)], "model": req.model or DEFAULT_MODEL}, config=cfg)
    return {"reply": res["messages"][-1].content}

@app.post("/rag/hq")
def rag_hq_endpoint(req: RagReq):
    g = rag_hq.build()
    cfg = {"configurable":{"thread_id": req.thread_id}}
    res = g.invoke({"messages":[HumanMessage(req.query)], "model": req.model or DEFAULT_MODEL}, config=cfg)
    return {"reply": res["messages"][-1].content}

@app.post("/rag/graph")
def rag_graph_endpoint(req: RagReq):
    g = rag_graph.build()
    cfg = {"configurable":{"thread_id": req.thread_id}}
    res = g.invoke({"messages":[HumanMessage(req.query)]}, config=cfg)
    return {"reply": res["messages"][-1].content}
