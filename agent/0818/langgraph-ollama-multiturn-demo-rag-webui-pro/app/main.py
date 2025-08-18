from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import UploadFile, File, Form
from typing import Optional
from app.rag.hybrid.hybrid_rag import hybrid_search, generate_answer
from app.rag.hybrid.index_hybrid import build_faiss as hybrid_build_faiss, build_bm25 as hybrid_build_bm25

from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage
from app.utils.model_tags import DEFAULT_MODEL, ROUTER_MODEL, GENERATOR_MODEL, OLLAMA_HOST

# 그래프 빌더들
from app.graphs import cp_mem_basic, cp_sqlite_persistent, subgraph_collect, subgraph_router_then_generate, multiturn_llm_branching, recover_node_try_except, recover_retry_fallback, checkpointers
# RAG graphs
from app.rag import basic_rag as rag_basic
from app.rag import hq_rag as rag_hq
from app.rag import graph_rag as rag_graph
from app.rag.chat import multiturn_basic as rag_mt_basic
from app.rag.chat import multiturn_hq as rag_mt_hq
from app.rag.chat import multiturn_graph as rag_mt_graph
from pydantic import BaseModel
from fastapi import UploadFile, File, Form
from typing import Optional
from app.rag.hybrid.hybrid_rag import hybrid_search, generate_answer
from app.rag.hybrid.index_hybrid import build_faiss as hybrid_build_faiss, build_bm25 as hybrid_build_bm25



app = FastAPI(title="LangGraph + Ollama Examples")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    if name == "checkpointers":
        return checkpointers.build()
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


class RagChatReq(BaseModel):
    message: str
    thread_id: str = "rag-chat"
    model: str | None = None

@app.post("/rag/chat/basic")
def rag_chat_basic(req: RagChatReq):
    g = rag_mt_basic.build()
    cfg = {"configurable":{"thread_id": req.thread_id}}
    res = g.invoke({"messages":[HumanMessage(req.message)], "model": req.model or DEFAULT_MODEL}, config=cfg)
    return {"reply": res["messages"][-1].content}

@app.post("/rag/chat/hq")
def rag_chat_hq(req: RagChatReq):
    g = rag_mt_hq.build()
    cfg = {"configurable":{"thread_id": req.thread_id}}
    res = g.invoke({"messages":[HumanMessage(req.message)], "model": req.model or DEFAULT_MODEL}, config=cfg)
    return {"reply": res["messages"][-1].content}

@app.post("/rag/chat/graph")
def rag_chat_graph(req: RagChatReq):
    g = rag_mt_graph.build()
    cfg = {"configurable":{"thread_id": req.thread_id}}
    res = g.invoke({"messages":[HumanMessage(req.message)]}, config=cfg)
    return {"reply": res["messages"][-1].content}


UPLOAD_DIR = "/data/uploads"

@app.post("/admin/upload")
async def upload(file: UploadFile = File(...)):
    import os, uuid
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    name = f"{uuid.uuid4().hex}_{file.filename}"
    path = os.path.join(UPLOAD_DIR, name)
    data = await file.read()
    with open(path, "wb") as f:
        f.write(data)
    return {"saved": path}

@app.post("/admin/reindex")
def reindex():
    # Rebuild basic/hq faiss and hybrid bm25/faiss from docs + uploads
    import shutil, glob
    docs_root = "/data/docs"
    up_root = "/data/uploads"
    temp_merge = "/data/_merged_docs"
    os.makedirs(temp_merge, exist_ok=True)
    # merge txt/md only
    def collect(src):
        if not os.path.exists(src): return
        for p in glob.glob(src + "/**/*.*", recursive=True):
            if p.lower().endswith((".txt",".md",".markdown",".text")):
                base = os.path.basename(p)
                shutil.copy2(p, os.path.join(temp_merge, base))
    collect(docs_root); collect(up_root)
    # Basic/HQ
    from app.rag.index_basic import build_index as build_basic
    build_basic(temp_merge, "/data/index/faiss_basic")
    build_basic(temp_merge, "/data/index/faiss_hq")
    # Hybrid
    hybrid_build_faiss(temp_merge, "/data/index/faiss_hybrid")
    hybrid_build_bm25(temp_merge, "/data/index/bm25.json")
    return {"status":"ok"}


class RagHybridReq(BaseModel):
    query: str
    thread_id: str = "rag-hybrid"
    model: str | None = None
    k: int = 6

@app.post("/rag/hybrid")
def rag_hybrid(req: RagHybridReq):
    k = max(1, min(10, req.k))
    hits = hybrid_search(req.query, k=k)
    snippets = [f"[{i+1}] {t}" for i,(t,s,src) in enumerate(hits)]
    model = req.model or DEFAULT_MODEL
    reply = generate_answer(req.query, snippets, model)
    return {"reply": reply, "hits": [{"score": s, "source": src} for (t,s,src) in hits]}


@app.get("/sse/rag")
async def sse_rag(mode: str, q: str, thread_id: str = "rag-stream", model: Optional[str] = None):
    from sse_starlette.sse import EventSourceResponse
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_community.chat_models import ChatOllama

    mdl = model or DEFAULT_MODEL

    # Build context by mode
    ctx = ""
    if mode == "basic":
        from langchain_community.vectorstores import FAISS
        from app.rag.embed import get_embedder
        vs = FAISS.load_local("/data/index/faiss_basic", embeddings=get_embedder(), allow_dangerous_deserialization=True)
        docs = vs.similarity_search(q, k=4)
        ctx = "\n\n".join([d.page_content.strip() for d in docs])
    elif mode == "hybrid":
        hits = hybrid_search(q, k=6)
        ctx = "\n\n".join([t for (t,s,src) in hits])
    else:
        # fallback to basic
        from langchain_community.vectorstores import FAISS
        from app.rag.embed import get_embedder
        vs = FAISS.load_local("/data/index/faiss_basic", embeddings=get_embedder(), allow_dangerous_deserialization=True)
        docs = vs.similarity_search(q, k=4)
        ctx = "\n\n".join([d.page_content.strip() for d in docs])

    def gen():
        sys = SystemMessage("한국어 RAG 어시스턴트. 컨텍스트를 근거로 점진적으로 답변을 스트리밍합니다.")
        user = HumanMessage(f"[질문]\n{q}\n\n[컨텍스트]\n{ctx}")
        chat = ChatOllama(model=mdl, temperature=0.2)
        # stream tokens
        for chunk in chat.stream([sys, user]):
            # chunk is an AIMessageChunk; get content delta
            part = getattr(chunk, "content", None)
            if part:
                yield {"event":"token", "data": part}
        yield {"event":"done", "data":"[DONE]"}
    return EventSourceResponse(gen())
