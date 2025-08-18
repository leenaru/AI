from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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


HYBRID_CONF = "/data/index/hybrid_config.json"
DELETED_JSON = "/data/index/deleted.json"

def _load_hybrid_conf():
    import os, json
    if os.path.exists(HYBRID_CONF):
        try:
            return json.load(open(HYBRID_CONF, "r", encoding="utf-8"))
        except Exception:
            pass
    return {"weight_vec": 0.5, "weight_bm25": 0.5, "fuse": "max", "rrf_k": 60}

@app.post("/admin/hybrid/tune")
def hybrid_tune(cfg: dict):
    os.makedirs(os.path.dirname(HYBRID_CONF), exist_ok=True)
    json.dump(cfg, open(HYBRID_CONF, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return {"saved": HYBRID_CONF, "config": cfg}

@app.get("/admin/hybrid/tune")
def hybrid_tune_get():
    return _load_hybrid_conf()

@app.post("/admin/delta/add_text")
def delta_add_text(payload: dict):
    # payload: { "text": "...", "source": "user_upload:note.txt" }
    text = payload.get("text","").strip()
    source = payload.get("source") or f"user_upload:{uuid.uuid4().hex}.txt"
    if not text:
        raise HTTPException(400, "text is required")
    # Update BM25 JSON
    import json, os
    bm25_json = "/data/index/bm25.json"
    os.makedirs("/data/index", exist_ok=True)
    if os.path.exists(bm25_json):
        obj = json.load(open(bm25_json, "r", encoding="utf-8"))
    else:
        obj = {"texts": [], "sources": []}
    obj["texts"].append(text)
    obj["sources"].append(source)
    json.dump(obj, open(bm25_json, "w", encoding="utf-8"), ensure_ascii=False)
    # Update FAISS (append)
    from app.rag.embed import get_embedder
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    docs = splitter.split_text(text)
    metadatas = [{"source": source} for _ in docs]
    try:
        vs = FAISS.load_local("/data/index/faiss_hybrid", embeddings=get_embedder(), allow_dangerous_deserialization=True)
        vs.add_texts(docs, metadatas=metadatas)
    except Exception:
        # If no index, create new
        vs = FAISS.from_texts(docs, embedding=get_embedder(), metadatas=metadatas)
    vs.save_local("/data/index/faiss_hybrid")
    return {"status":"ok", "chunks": len(docs), "source": source}

@app.post("/admin/delta/delete")
def delta_delete(payload: dict):
    # payload: { "source": "user_upload:note.txt" }
    src = payload.get("source")
    if not src:
        raise HTTPException(400, "source is required")
    os.makedirs(os.path.dirname(DELETED_JSON), exist_ok=True)
    data = []
    if os.path.exists(DELETED_JSON):
        try:
            data = json.load(open(DELETED_JSON, "r", encoding="utf-8"))
        except Exception:
            data = []
    if src not in data:
        data.append(src)
    json.dump(data, open(DELETED_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # BM25 json에서 즉시 제거
    bm25_json = "/data/index/bm25.json"
    if os.path.exists(bm25_json):
        obj = json.load(open(bm25_json, "r", encoding="utf-8"))
        texts, sources = obj.get("texts",[]), obj.get("sources",[])
        filt = [(t,s) for (t,s) in zip(texts, sources) if s != src]
        obj["texts"] = [t for t,_ in filt]
        obj["sources"] = [s for _,s in filt]
        json.dump(obj, open(bm25_json, "w", encoding="utf-8"), ensure_ascii=False)
    return {"deleted_marked": src, "note":"FAISS는 즉시 삭제가 어려워 compact 필요"}

@app.post("/admin/compact")
def compact():
    # 완전 재색인: /data/docs + /data/uploads - deleted.json
    import shutil, glob
    docs_root = "/data/docs"
    up_root = "/data/uploads"
    temp_merge = "/data/_merged_docs"
    os.makedirs(temp_merge, exist_ok=True)
    for p in glob.glob(temp_merge + "/*"):
        try:
            os.remove(p)
        except:
            pass
    def collect(src, excluded):
        if not os.path.exists(src): return
        for p in glob.glob(src + "/**/*.*", recursive=True):
            if p.lower().endswith((".txt",".md",".markdown",".text")) and p not in excluded:
                base = os.path.basename(p)
                shutil.copy2(p, os.path.join(temp_merge, base))
    deleted = set(json.load(open(DELETED_JSON, "r", encoding="utf-8"))) if os.path.exists(DELETED_JSON) else set()
    collect(docs_root, deleted); collect(up_root, deleted)
    # Basic/HQ
    from app.rag.index_basic import build_index as build_basic
    build_basic(temp_merge, "/data/index/faiss_basic")
    build_basic(temp_merge, "/data/index/faiss_hq")
    # Hybrid
    from app.rag.hybrid.index_hybrid import build_faiss as hybrid_build_faiss, build_bm25 as hybrid_build_bm25
    hybrid_build_faiss(temp_merge, "/data/index/faiss_hybrid")
    hybrid_build_bm25(temp_merge, "/data/index/bm25.json")
    return {"status":"ok","excluded":len(deleted)}


@app.websocket("/ws/rag")
async def ws_rag(websocket: WebSocket, mode: str = "basic", model: Optional[str] = None):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            q = msg
            # 컨텍스트/출처 수집
            hits = []
            if mode == "hybrid":
                from app.rag.hybrid.hybrid_rag import hybrid_search
                hits = hybrid_search(q, k=6)
                ctx = "\n\n".join([t for (t,s,src) in hits])
            else:
                from langchain_community.vectorstores import FAISS
                from app.rag.embed import get_embedder
                vs = FAISS.load_local("/data/index/faiss_basic", embeddings=get_embedder(), allow_dangerous_deserialization=True)
                docs = vs.similarity_search(q, k=4)
                hits = [(d.page_content.strip(), 1.0, d.metadata.get("source","")) for d in docs]
                ctx = "\n\n".join([d.page_content.strip() for d in docs])
            await websocket.send_json({"type":"ctx", "hits":[{"text":t[:500], "score":s, "source":src} for (t,s,src) in hits]})
            # 스트리밍
            from langchain_community.chat_models import ChatOllama
            from langchain_core.messages import SystemMessage, HumanMessage
            mdl = model or DEFAULT_MODEL
            sys = SystemMessage("한국어 RAG 어시스턴트. 컨텍스트를 근거로 스트리밍합니다.")
            user = HumanMessage(f"[질문]\n{q}\n\n[컨텍스트]\n{ctx}")
            chat = ChatOllama(model=mdl, temperature=0.2)
            for chunk in chat.stream([sys, user]):
                part = getattr(chunk, "content", None)
                if part:
                    await websocket.send_json({"type":"token", "delta": part})
            await websocket.send_json({"type":"done"})
    except WebSocketDisconnect:
        return


@app.get("/source")
def get_source(source: str):
    """원문 뷰어: 파일 경로면 디스크에서 읽고, 그렇지 않으면 BM25 JSON에서 매칭된 텍스트 반환."""
    import os, json
    # Try file path
    if os.path.exists(source) and os.path.isfile(source):
        try:
            with open(source, "r", encoding="utf-8", errors="ignore") as f:
                return {"source": source, "text": f.read()}
        except Exception as e:
            raise HTTPException(500, f"cannot read file: {e}")
    # Try bm25 json mapping
    bm25_json = "/data/index/bm25.json"
    if os.path.exists(bm25_json):
        obj = json.load(open(bm25_json, "r", encoding="utf-8"))
        texts, sources = obj.get("texts",[]), obj.get("sources",[])
        for t,s in zip(texts, sources):
            if s == source:
                return {"source": s, "text": t}
    # Not found
    raise HTTPException(404, f"source not found: {source}")


FAISS_IDS = "/data/index/faiss_ids.json"

def _load_ids_map():
    if os.path.exists(FAISS_IDS):
        try:
            return json.load(open(FAISS_IDS, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_ids_map(m):
    os.makedirs(os.path.dirname(FAISS_IDS), exist_ok=True)
    json.dump(m, open(FAISS_IDS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

@app.post("/admin/compact_one")
def compact_one(payload: dict):
    """특정 source만 FAISS에서 삭제 후, 현재 파일/텍스트로 재인덱싱."""
    from app.rag.embed import get_embedder
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    source = payload.get("source")
    text = payload.get("text")  # 선택: 직접 제공하면 파일 대신 사용
    if not source:
        raise HTTPException(400, "source is required")

    ids_map = _load_ids_map()
    try:
        vs = FAISS.load_local("/data/index/faiss_hybrid", embeddings=get_embedder(), allow_dangerous_deserialization=True)
    except Exception as e:
        raise HTTPException(500, f"faiss load error: {e}")

    # delete old ids if known
    ids = ids_map.get(source, [])
    if ids:
        try:
            vs.delete(ids)
        except Exception:
            pass

    # fetch fresh text
    if text is None:
        # read from file or bm25 map
        try:
            if os.path.exists(source):
                text = open(source, "r", encoding="utf-8", errors="ignore").read()
            else:
                obj = json.load(open("/data/index/bm25.json", "r", encoding="utf-8"))
                for t,s in zip(obj.get("texts",[]), obj.get("sources",[])):
                    if s == source:
                        text = t; break
        except Exception as e:
            raise HTTPException(500, f"cannot read source: {e}")
    if not text:
        raise HTTPException(404, "no text for source")

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = splitter.split_text(text)
    new_ids = [f"{source}::" + hashlib.sha1((source + '|' + c).encode('utf-8')).hexdigest()[:16] for c in chunks]
    metadatas = [{"source": source} for _ in chunks]
    try:
        vs.add_texts(chunks, metadatas=metadatas, ids=new_ids)
    except TypeError:
        # older langchain may not support ids kw; fallback add then rebuild map unknown
        vs.add_texts(chunks, metadatas=metadatas)
        new_ids = []

    vs.save_local("/data/index/faiss_hybrid")
    ids_map[source] = new_ids
    _save_ids_map(ids_map)

    # also refresh BM25 record: remove previous + append
    bm25_json = "/data/index/bm25.json"
    try:
        if os.path.exists(bm25_json):
            obj = json.load(open(bm25_json, "r", encoding="utf-8"))
            texts, sources = obj.get("texts",[]), obj.get("sources",[])
            filt = [(t,s) for (t,s) in zip(texts, sources) if s != source]
            obj["texts"] = [t for t,_ in filt] + [text]
            obj["sources"] = [s for _,s in filt] + [source]
        else:
            obj = {"texts":[text], "sources":[source]}
        json.dump(obj, open(bm25_json, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception as e:
        raise HTTPException(500, f"bm25 update error: {e}")

    return {"status":"ok", "chunks": len(chunks), "source": source, "faiss_ids": len(new_ids)}


class ProfileReq(BaseModel):
    prompt: str
    models: list[str]
    mode: str | None = "basic"
    k: int = 4

@app.post("/profile/stream")
def profile_stream(req: ProfileReq):
    """모델별 스트리밍 성능(첫 토큰 지연/총 시간/토큰수/tps) 측정."""
    import time
    from langchain_community.chat_models import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage
    # build context
    ctx = ""
    if req.mode == "basic":
        from langchain_community.vectorstores import FAISS
        from app.rag.embed import get_embedder
        try:
            vs = FAISS.load_local("/data/index/faiss_basic", embeddings=get_embedder(), allow_dangerous_deserialization=True)
            docs = vs.similarity_search(req.prompt, k=req.k)
            ctx = "\n\n".join([d.page_content.strip() for d in docs])
        except Exception:
            ctx = ""
    elif req.mode == "hybrid":
        from app.rag.hybrid.hybrid_rag import hybrid_search
        hits = hybrid_search(req.prompt, k=req.k)
        ctx = "\n\n".join([t for (t,s,src) in hits])
    else:
        ctx = ""
    results = []
    for model in req.models:
        sys = SystemMessage("한국어 RAG 스트리밍 프로파일링. 간단히 답하세요.")
        user = HumanMessage(f"[Q]\n{req.prompt}\n\n[CTX]\n{ctx}")
        chat = ChatOllama(model=model, temperature=0.2)
        t0 = time.time()
        first = None
        tokens = 0
        for chunk in chat.stream([sys, user]):
            if first is None:
                first = time.time()
            part = getattr(chunk, "content", "")
            if part:
                tokens += len(part.split())
        t1 = time.time()
        results.append({
            "model": model,
            "first_token_ms": int(((first or t1) - t0)*1000),
            "total_ms": int((t1 - t0)*1000),
            "tokens": tokens,
            "tps": round(tokens / max(1e-6, (t1 - (first or t0))), 2) if tokens>0 else 0.0
        })
    return {"prompt": req.prompt, "mode": req.mode, "results": results}
