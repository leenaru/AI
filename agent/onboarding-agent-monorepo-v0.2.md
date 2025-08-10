# Onboarding Agent – Monorepo (v0.2)
**What’s inside**  
- `server/`: FastAPI + LangGraph + Ollama, HQ‑RAG (in‑memory) with optional **Qdrant** vector search, policy overlays, appliance scenario, tickets + RAG ingestion.
- `gui/`: React Flow graph editor (YAML ⇄ Graph).
- `clients/`: Android / iOS action orchestrator samples.

**Highlights**  
- Client action loop: `OPEN_CAMERA`, `REQUEST_INFO`, `CREATE_TICKET`.
- Appliance flows: install / usage / troubleshoot (accuracy‑first, escalation) / warranty / service.
- Operator loop: unresolved issues → ticket → resolution → **RAG update**.
- Qdrant integration: turn on by setting `QDRANT_URL` in `.env` or via `docker compose`.
- Graph GUI: edit YAML visually, export back to YAML.

## Quickstart

### Server (Python)
```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# (Optional) appliance graph
# echo 'GRAPH_FILE=graphs/appliance.yaml' >> .env
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### Server (Docker: Ollama + Qdrant included)
```bash
cd server
docker compose up -d
docker exec -it ollama ollama pull llama3:instruct
# API: http://localhost:8080
```

**Test a troubleshooting flow**
```bash
curl -s http://localhost:8080/v1/chat -H 'Content-Type: application/json' -d '{
  "text": "에어컨이 E5 에러가 떠요",
  "lang": "ko-KR",
  "product": {"brand":"Acme","model":"AC-1234"},
  "policy_context": {"country":"KR"}
}'
```

### GUI
```bash
cd gui
npm i
npm run dev
# http://localhost:5173
```

---
## Files in this bundle

- `README.md`- `clients/android/ActionOrchestrator.kt`- `clients/ios/ActionOrchestrator.swift`- `gui/index.html`- `gui/package.json`- `gui/src/App.tsx`- `gui/src/GraphEditor.tsx`- `gui/src/dsl.ts`- `gui/src/main.tsx`- `gui/tsconfig.json`- `gui/vite.config.ts`- `server/.env.example`- `server/Dockerfile`- `server/README.md`- `server/app/api/v1/chat.py`- `server/app/api/v1/events.py`- `server/app/api/v1/rag_admin.py`- `server/app/api/v1/tickets.py`- `server/app/core/config.py`- `server/app/core/logging.py`- `server/app/main.py`- `server/app/models/schemas.py`- `server/app/services/langgraph/graph_runner.py`- `server/app/services/langgraph/nodes.py`- `server/app/services/model_router.py`- `server/app/services/ollama_client.py`- `server/app/services/policy/enforcer.py`- `server/app/services/policy/overlay_loader.py`- `server/app/services/rag/__init__.py`- `server/app/services/rag/embeddings.py`- `server/app/services/rag/indexer_qdrant.py`- `server/app/services/rag/memory_store.py`- `server/app/services/rag/service.py`- `server/app/services/rag/service_qdrant.py`- `server/app/utils/pii.py`- `server/docker-compose.yml`- `server/graphs/appliance.yaml`- `server/graphs/core.yaml`- `server/graphs/overlays/country/kr.yaml`- `server/graphs/policies/std-privacy.yaml`- `server/requirements.txt`- `server/tests/test_smoke.py`
---
### README.md
```md
# Onboarding Agent – Server + GUI Skeleton (v0.2)

This bundle contains:
- **server/**: FastAPI + LangGraph runtime + Ollama model router + HQ‑RAG (with optional **Qdrant**).
- **gui/**: React Flow graph editor (YAML ⇄ GUI).
- **clients/**: Android/iOS **action orchestrator** samples.

## Quickstart (Server)

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# (Optional) set GRAPH_FILE=graphs/appliance.yaml to try appliance scenario
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### Docker (Ollama + Qdrant included)
```bash
cd server
docker compose up -d
docker exec -it ollama ollama pull llama3:instruct
# API: http://localhost:8080
```

### Endpoints
- `POST /v1/chat` – main chat/intent (actions/citations)
- `POST /v1/events/camera` – camera proactive event ingest
- `POST /v1/tickets` / `PATCH /v1/tickets/{id}/resolve` – escalation + *RAG update* by agents
- `POST /v1/rag/ingest` – manual knowledge ingestion (operators)
- `GET /healthz`

### Curl demo (Troubleshooting → Escalation → Operator Resolve → RAG update)
```bash
curl -s http://localhost:8080/v1/chat -H 'Content-Type: application/json' -d '{
  "text": "에어컨이 E5 에러가 떠요",
  "lang": "ko-KR",
  "product": {"brand":"Acme","model":"AC-1234"},
  "policy_context": {"country":"KR"}
}'
```

## Quickstart (GUI)
```bash
cd gui
npm i
npm run dev
# http://localhost:5173
```

## Notes
- To switch to the **appliance** graph: set `GRAPH_FILE=graphs/appliance.yaml` in `.env`.
- If **Qdrant** is up (via docker-compose) the RAG automatically uses it, otherwise in‑memory.

```
### clients/android/ActionOrchestrator.kt
```kotlin
// Minimal orchestrator sample for Android (Kotlin)
sealed class ClientAction {
    data class OpenCamera(val mode: String, val timeoutMs: Long?): ClientAction()
    data class RequestInfo(val fields: List<String>): ClientAction()
}
data class ServerAction(val name: String, val params: Map<String, Any?>)
data class ActionEnvelope(
    val clientActions: List<ClientAction> = emptyList(),
    val serverActions: List<ServerAction> = emptyList()
)
class ActionOrchestrator {
    suspend fun handle(envelope: ActionEnvelope) {
        // TODO: implement API calls, permission checks, CameraX capture, and uplink
    }
}

```
### clients/ios/ActionOrchestrator.swift
```swift
// Minimal orchestrator sample for iOS (SwiftUI)
import SwiftUI
import AVFoundation

enum ClientAction {
    case openCamera(mode: String, timeoutMs: Int?)
    case requestInfo(fields: [String])
}
struct ActionEnvelope {
    var clientActions: [ClientAction] = []
    var serverActions: [(name: String, params: [String:Any])] = []
}
final class ActionOrchestrator: ObservableObject {
    @Published var showCamera = false
    func handle(_ env: ActionEnvelope) {
        // TODO: implement URLSession calls, permission checks and camera sheet
    }
}

```
### gui/index.html
```html
<!doctype html>
<html>
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Graph GUI</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>

```
### gui/package.json
```json
{
  "name": "graph-gui",
  "private": true,
  "version": "0.1.0",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview --port 5174"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "reactflow": "^11.10.0",
    "js-yaml": "^4.1.0"
  },
  "devDependencies": {
    "typescript": "^5.5.4",
    "vite": "^5.2.0",
    "@types/react": "^18.2.56",
    "@types/react-dom": "^18.2.19"
  }
}

```
### gui/src/App.tsx
```tsx
import React, { useState } from 'react'
import GraphEditor from './GraphEditor'
import { parseYaml, toYaml, Dsl } from './dsl'

const initText = `graph:
  id: core-v1
  nodes:
    - id: intent_router
      type: router
    - id: rag_answer
      type: llm
  edges:
    - from: intent_router
      to: rag_answer
`

export default function App(){
  const [yamlText, setYamlText] = useState(initText)
  const [dsl, setDsl] = useState<Dsl>(parseYaml(initText))
  return (
    <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, padding:16}}>
      <div>
        <h3>YAML</h3>
        <textarea value={yamlText} onChange={(e)=>{ setYamlText(e.target.value); setDsl(parseYaml(e.target.value)) }} style={{width:'100%', height:'80vh'}}/>
      </div>
      <div>
        <h3>Graph</h3>
        <GraphEditor dsl={dsl} onChange={(d)=> setYamlText(toYaml(d)) }/>
        <h4>Exported YAML</h4>
        <pre style={{whiteSpace:'pre-wrap', background:'#fafafa', padding:8, borderRadius:8}}>{yamlText}</pre>
      </div>
    </div>
  )
}

```
### gui/src/GraphEditor.tsx
```tsx
import React, { useCallback, useState } from 'react'
import ReactFlow, { Background, Controls, addEdge, Connection, Edge, Node } from 'reactflow'
import 'reactflow/dist/style.css'
import { Dsl } from './dsl'

export default function GraphEditor({ dsl, onChange }:{ dsl:Dsl, onChange:(d:Dsl)=>void }){
  const [nodes, setNodes] = useState<Node[]>(dsl.graph.nodes.map((n,i)=>({id:n.id, position:{x:120*i,y:100}, data:{label:`${n.id} (${n.type})`}})))
  const [edges, setEdges] = useState<Edge[]>(dsl.graph.edges.map(e=>({id:`${e.from}->${e.to}`, source:e.from, target:e.to})))

  const onConnect = useCallback((c:Connection)=> setEdges(eds=>addEdge(c, eds)), [])
  const addNode = () => {
    const id = `node_${nodes.length+1}`
    const n:Node = {id, position:{x:50,y:50}, data:{label:`${id} (llm)`}}
    setNodes(ns=>[...ns,n])
  }
  const exportDsl = () => {
    const d:Dsl = {
      graph:{
        id: dsl.graph.id,
        nodes: nodes.map(n=>({id:n.id, type: String(n.data?.label).split('(')[1]?.replace(')','') || 'llm'})),
        edges: edges.map(e=>({from:e.source as string, to:e.target as string}))
      },
      metadata: dsl.metadata
    }
    onChange(d)
  }

  return (
    <div style={{height:'80vh', border:'1px solid #eee', borderRadius:16, padding:8}}>
      <div style={{display:'flex', gap:8, marginBottom:8}}>
        <button onClick={addNode}>노드 추가</button>
        <button onClick={exportDsl}>YAML 업데이트</button>
      </div>
      <ReactFlow nodes={nodes} edges={edges} onConnect={onConnect} fitView>
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  )
}

```
### gui/src/dsl.ts
```ts
import yaml from 'js-yaml'

export type Dsl = {
  graph: {
    id: string,
    nodes: {id: string, type: string}[],
    edges: {from: string, to: string}[],
  },
  metadata?: any
}

export function parseYaml(text: string): Dsl {
  return yaml.load(text) as Dsl
}

export function toYaml(dsl: Dsl): string {
  return yaml.dump(dsl)
}

```
### gui/src/main.tsx
```tsx
import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'

const el = document.getElementById('root')!
createRoot(el).render(<App />)

```
### gui/tsconfig.json
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "jsx": "react-jsx",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "strict": true,
    "skipLibCheck": true
  },
  "include": ["src"]
}

```
### gui/vite.config.ts
```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173
  }
})

```
### server/.env.example
```
APP_ENV=dev
API_PORT=8080

# Model routing
DEFAULT_MODEL=llama3:instruct
OLLAMA_BASE_URL=http://ollama:11434

# RAG
RAG_MIN_CITATIONS=2
RAG_FRESHNESS_DAYS=60

# Policy
POLICY_OVERLAYS=country:KR,policy:std-privacy
PII_MASKING=strict

# Graph (core or appliance)
GRAPH_FILE=graphs/core.yaml

# Qdrant (optional)
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=kb_docs
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

```
### server/Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

```
### server/README.md
```md
# Server – LangGraph + Ollama (with optional Qdrant)

See top-level README for quickstart. This service exposes FastAPI endpoints for chat, events, tickets and RAG ingestion.

```
### server/app/api/v1/chat.py
```python
from fastapi import APIRouter
from app.models.schemas import ChatRequest, Answer, Citation
from app.services.langgraph.graph_runner import get_runner

router = APIRouter()

@router.post("/chat", response_model=Answer)
async def chat(req: ChatRequest):
    runner = get_runner()
    state = req.model_dump()
    result = await runner.run(state)
    return Answer(
        answer=result.get("answer", ""),
        citations=[Citation(**c) for c in result.get("citations", [])],
        actions=result.get("actions")
    )

```
### server/app/api/v1/events.py
```python
from fastapi import APIRouter
from app.models.schemas import CameraEvent, Answer

router = APIRouter()

@router.post("/events/camera", response_model=Answer)
async def camera_event(ev: CameraEvent):
    actions = []
    if ev.event_type == "document_present" and ev.score >= 0.8:
        actions.append({
            "type": "client_action",
            "action": "OPEN_CAMERA",
            "params": {"mode": "document"},
            "safety": {"requires_user_confirm": True, "reason": "문서 촬영"}
        })
    return Answer(answer="이벤트 처리됨", citations=[], actions=actions)

```
### server/app/api/v1/rag_admin.py
```python
from fastapi import APIRouter
from app.models.schemas import RAGDocIngest
from app.services.rag import rag

router = APIRouter()

@router.post("/rag/ingest")
async def rag_ingest(doc: RAGDocIngest):
    saved = rag.upsert([{
        "title": doc.title,
        "content": doc.content,
        "url": doc.url,
        "tags": doc.tags,
        "products": doc.products,
        "date": doc.date
    }]) if hasattr(rag, "upsert") and callable(getattr(rag, "upsert")) and "Qdrant" in rag.__class__.__name__ else rag.upsert(
        title=doc.title, content=doc.content, url=doc.url, tags=doc.tags, products=doc.products, date=doc.date
    )
    return {"ok": True, "doc": saved}

```
### server/app/api/v1/tickets.py
```python
from fastapi import APIRouter, HTTPException
from typing import Dict
from app.models.schemas import Ticket, TicketResolve
from app.services.rag import rag

router = APIRouter()

TICKETS: Dict[str, Ticket] = {}

@router.post("/tickets")
async def create_ticket(t: Ticket):
    t.id = t.id or str(len(TICKETS) + 1)
    TICKETS[t.id] = t
    return {"ok": True, "ticket": t}

@router.get("/tickets/{tid}")
async def get_ticket(tid: str):
    if tid not in TICKETS:
        raise HTTPException(404, "not found")
    return TICKETS[tid]

@router.patch("/tickets/{tid}/resolve")
async def resolve_ticket(tid: str, body: TicketResolve):
    if tid not in TICKETS:
        raise HTTPException(404, "not found")
    t = TICKETS[tid]
    t.status = "resolved"
    t.notes.append(body.resolution)
    # upsert to RAG
    if hasattr(rag, "upsert") and "Qdrant" in rag.__class__.__name__:
        rag.upsert([{
            "title": f"해결: {t.symptom}",
            "content": body.resolution,
            "url": body.url,
            "tags": body.tags,
            "products": [t.product.model] if (t.product and t.product.model) else body.products
        }])
    else:
        rag.upsert(title=f"해결: {t.symptom}", content=body.resolution, url=body.url, tags=body.tags,
                   products=[t.product.model] if (t.product and t.product.model) else body.products)
    return {"ok": True, "ticket": t}

```
### server/app/core/config.py
```python
from pydantic import BaseSettings, Field
from typing import List
import os

class Settings(BaseSettings):
    app_env: str = Field(default=os.getenv("APP_ENV", "dev"))
    api_port: int = Field(default=int(os.getenv("API_PORT", 8080)))

    default_model: str = Field(default=os.getenv("DEFAULT_MODEL", "llama3:instruct"))
    ollama_base_url: str = Field(default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    rag_min_citations: int = Field(default=int(os.getenv("RAG_MIN_CITATIONS", 2)))
    rag_freshness_days: int = Field(default=int(os.getenv("RAG_FRESHNESS_DAYS", 60)))

    policy_overlays: List[str] = Field(default_factory=lambda: os.getenv("POLICY_OVERLAYS", "").split(",") if os.getenv("POLICY_OVERLAYS") else [])
    pii_masking: str = Field(default=os.getenv("PII_MASKING", "strict"))

    graph_file: str = Field(default=os.getenv("GRAPH_FILE", "graphs/core.yaml"))

    # Qdrant
    qdrant_url: str = Field(default=os.getenv("QDRANT_URL", ""))
    qdrant_collection: str = Field(default=os.getenv("QDRANT_COLLECTION", "kb_docs"))
    embed_model: str = Field(default=os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))

    class Config:
        env_file = ".env"

settings = Settings()

```
### server/app/core/logging.py
```python
import logging, sys, json, time

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

def setup_logging(level="INFO"):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

```
### server/app/main.py
```python
from fastapi import FastAPI
from app.core.logging import setup_logging
from app.core.config import settings
from app.api.v1 import chat, events, rag_admin, tickets

setup_logging("INFO")
app = FastAPI(title="On-device + Server AI – API", version="0.2.0")

app.include_router(chat.router, prefix="/v1")
app.include_router(events.router, prefix="/v1")
app.include_router(rag_admin.router, prefix="/v1")
app.include_router(tickets.router, prefix="/v1")

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "env": settings.app_env}

```
### server/app/models/schemas.py
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class DeviceContext(BaseModel):
    platform: Optional[str] = None
    network: Optional[str] = None
    battery: Optional[float] = None
    permissions: Optional[Dict[str, bool]] = None

class PolicyContext(BaseModel):
    country: Optional[str] = None
    age_mode: Optional[str] = "adult"

class MultimodalPayload(BaseModel):
    image_refs: Optional[List[str]] = None
    audio_ref: Optional[str] = None

class ProductInfo(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    serial: Optional[str] = None

class TroubleshootContext(BaseModel):
    symptom: Optional[str] = None
    severity: Optional[str] = None  # info|warn|error|critical

class ChatRequest(BaseModel):
    type: str = Field("graph_query")
    lang: str = Field("ko-KR")
    text: Optional[str] = None
    channels: List[str] = ["text"]
    multimodal: Optional[MultimodalPayload] = None
    device_context: Optional[DeviceContext] = None
    policy_context: Optional[PolicyContext] = None
    overlays: Optional[List[str]] = None
    product: Optional[ProductInfo] = None
    troubleshoot: Optional[TroubleshootContext] = None

class Citation(BaseModel):
    title: str
    url: str | None = None
    date: Optional[str] = None

class Answer(BaseModel):
    answer: str
    citations: List[Citation] = []
    actions: Optional[List[Dict[str, Any]]] = None

class CameraEvent(BaseModel):
    user_id: str
    event_type: str
    score: float
    meta: Dict[str, Any] = {}

class RAGDocIngest(BaseModel):
    title: str
    url: Optional[str] = None
    content: str
    date: Optional[str] = None
    tags: List[str] = []
    products: List[str] = []

class Ticket(BaseModel):
    id: Optional[str] = None
    user_id: str
    product: Optional[ProductInfo] = None
    symptom: str
    status: str = "open"
    notes: List[str] = []

class TicketResolve(BaseModel):
    resolution: str
    add_to_rag: bool = True
    tags: List[str] = ["troubleshooting","verified"]
    url: Optional[str] = None
    products: List[str] = []

```
### server/app/services/langgraph/graph_runner.py
```python
from typing import Dict, Any
import yaml
from app.core.config import settings
from app.services.policy.overlay_loader import load_overlays
from app.services.policy.enforcer import PolicyEnforcer
from .nodes import (
    intent_router_node, rag_answer_node, llm_answer_node, action_suggester_node,
    require_product_context, install_flow_node, usage_flow_node,
    troubleshoot_flow_node, warranty_flow_node, service_flow_node
)

try:
    from langgraph.graph import StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

class GraphRunner:
    def __init__(self, graph_file: str | None = None, overlays: list[str] | None = None):
        self.graph_cfg = yaml.safe_load(open(graph_file or settings.graph_file, "r", encoding="utf-8").read())
        self.overlays = overlays or settings.policy_overlays
        self.rules = load_overlays(self.overlays).get("rules", {})
        self.enforcer = PolicyEnforcer(self.rules)
        if LANGGRAPH_AVAILABLE:
            self._build_langgraph()

    def _build_langgraph(self):
        sg = StateGraph(dict)
        sg.add_node("intent_router", intent_router_node)
        sg.add_node("rag_answer", rag_answer_node)
        sg.add_node("llm_answer", llm_answer_node)
        sg.add_node("require_product", require_product_context)
        sg.add_node("install_flow", install_flow_node)
        sg.add_node("usage_flow", usage_flow_node)
        sg.add_node("troubleshoot_flow", troubleshoot_flow_node)
        sg.add_node("warranty_flow", warranty_flow_node)
        sg.add_node("service_flow", service_flow_node)
        sg.add_node("action_suggester", action_suggester_node)

        sg.set_entry_point("intent_router")
        # fan-out edges
        for tgt in ["rag_answer","llm_answer","require_product","warranty_flow","service_flow","install_flow","usage_flow","troubleshoot_flow"]:
            sg.add_edge("intent_router", tgt)
        # terminal mapping to action suggester
        for node in ["rag_answer","llm_answer","install_flow","usage_flow","troubleshoot_flow","warranty_flow","service_flow","require_product"]:
            sg.add_edge(node, "action_suggester")

        self.app = sg.compile()

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("policy_context", {}).get("country"):
            state["policy_country"] = state["policy_context"]["country"]
        state["lang"] = state.get("lang") or "ko-KR"

        if LANGGRAPH_AVAILABLE:
            result = await self.app.ainvoke(state)
        else:
            # minimal manual path
            result = await intent_router_node(state)
            route = result.get("route")
            if route == "rag_answer":
                result = await rag_answer_node(result)
            elif route == "install_flow":
                result = await require_product_context(result)
                result = await install_flow_node(result)
            elif route == "usage_flow":
                result = await require_product_context(result)
                result = await usage_flow_node(result)
            elif route == "troubleshoot_flow":
                result = await require_product_context(result)
                result = await troubleshoot_flow_node(result)
            elif route == "warranty_flow":
                result = await warranty_flow_node(result)
            elif route == "service_flow":
                result = await service_flow_node(result)
            else:
                result = await llm_answer_node(result)
            result = await action_suggester_node(result)

        ans, cites = self.enforcer.enforce(result.get("answer", ""), result.get("citations", []))
        result["answer"], result["citations"] = ans, cites
        return result

def get_runner() -> GraphRunner:
    return GraphRunner(graph_file=settings.graph_file, overlays=settings.policy_overlays)

```
### server/app/services/langgraph/nodes.py
```python
from typing import Dict, Any
from app.services.rag import rag
from app.services.model_router import choose_model
from app.services.ollama_client import ollama

async def intent_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    text = (state.get("text") or "").lower()
    if any(k in text for k in ["설치", "연결", "추가", "페어링"]):
        state["route"] = "install_flow"
    elif any(k in text for k in ["등록", "정품"]):
        state["route"] = "warranty_flow"
    elif any(k in text for k in ["사용법", "어떻게", "기능"]):
        state["route"] = "usage_flow"
    elif any(k in text for k in ["고장", "에러", "오류", "안됨", "소음", "누수"]):
        state["route"] = "troubleshoot_flow"
    elif any(k in text for k in ["as", "수리", "방문", "접수"]):
        state["route"] = "service_flow"
    elif any(k in text for k in ["무엇", "언제", "어디", "정의", "설명", "근거"]):
        state["route"] = "rag_answer"
    else:
        state["route"] = "llm_answer"
    return state

async def require_product_context(state: Dict[str, Any]) -> Dict[str, Any]:
    prod = state.get("product") or {}
    if not prod.get("model"):
        state.setdefault("actions", []).append({
            "type": "client_action",
            "action": "REQUEST_INFO",
            "params": {"fields": ["brand","model","serial"]},
            "safety": {"requires_user_confirm": False}
        })
        state["answer"] = "제품 모델명/시리얼을 알려주세요. 필요 시 제품 라벨 촬영을 도와드릴게요."
    return state

async def rag_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("text") or ""
    retrieved = rag.retrieve(query)
    cites = rag.enforce_citations("", retrieved)
    model = choose_model(state.get("lang"), state.get("policy_country"))
    messages = [{"role": "user", "content": f"질문: {query}\n참고자료 제목: {[d.get('title') for d in retrieved]}\n간결히 요약 답변."}]
    llm = await ollama.chat(model=model, messages=messages)
    state["answer"] = llm.get("message", {}).get("content", "요약을 생성하지 못했습니다.")
    state["citations"] = cites
    return state

async def llm_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    model = choose_model(state.get("lang"), state.get("policy_country"))
    messages = [{"role": "user", "content": state.get("text") or ""}]
    llm = await ollama.chat(model=model, messages=messages)
    state["answer"] = llm.get("message", {}).get("content", "응답을 생성하지 못했습니다.")
    state["citations"] = []
    return state

async def install_flow_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prod_model = (state.get('product') or {}).get('model', '')
    q = f"설치 가이드 {prod_model}"
    retrieved = rag.retrieve(q, products=[prod_model] if prod_model else None, tags=["guide","install"])
    cites = rag.enforce_citations("", retrieved)
    model = choose_model(state.get("lang"), state.get("policy_country"))
    messages = [{"role":"user","content": f"다음 자료 기반으로 설치 절차를 단계별 bullet로 간결히 요약: {[d.get('title') for d in retrieved]}"}]
    llm = await ollama.chat(model=model, messages=messages)
    state["answer"] = llm.get("message",{}).get("content","설치 절차를 찾지 못했습니다.")
    state["citations"] = cites
    return state

async def usage_flow_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prod_model = (state.get('product') or {}).get('model', '')
    q = f"사용법 {state.get('text','')} {prod_model}"
    retrieved = rag.retrieve(q, products=[prod_model] if prod_model else None, tags=["guide","faq"])
    cites = rag.enforce_citations("", retrieved)
    model = choose_model(state.get("lang"), state.get("policy_country"))
    messages = [{"role":"user","content": f"아래 문서를 근거로 사용법을 요약: {[d.get('title') for d in retrieved]}"}]
    llm = await ollama.chat(model=model, messages=messages)
    state["answer"] = llm.get("message",{}).get("content","도움을 찾지 못했습니다.")
    state["citations"] = cites
    return state

async def troubleshoot_flow_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prod_model = (state.get('product') or {}).get('model')
    symptom = (state.get('troubleshoot') or {}).get('symptom') or state.get('text','')

    if not prod_model:
        state.setdefault("actions", []).append({
            "type":"client_action","action":"OPEN_CAMERA","params":{"mode":"label"},
            "safety":{"requires_user_confirm": True, "reason":"모델 라벨 인식"}
        })
    retrieved = rag.retrieve(symptom, products=[prod_model] if prod_model else None, tags=["troubleshooting","runbook"])
    cites = rag.enforce_citations("", retrieved)

    confidence = 0
    if any("verified" in (d.get("tags") or []) for d in retrieved): confidence += 1
    if prod_model and any(prod_model in (d.get("products") or []) for d in retrieved): confidence += 1
    if len(retrieved) >= 2: confidence += 1

    model = choose_model(state.get("lang"), state.get("policy_country"))
    prompt = (
        "당신은 가전 A/S 기술 문서 기반의 수리 가이드입니다.\n"
        "응답 형식:\n"
        "1) 원인 후보(우선순위)\n2) 안전 경고\n3) 단계별 조치(체크리스트)\n4) 검증 방법\n"
        "반드시 근거 문서의 범위를 벗어나지 말고, 모호하면 '확신 부족'으로 표시하세요."
    )
    titles = [d.get('title') for d in retrieved]
    messages = [
        {"role":"system","content": prompt},
        {"role":"user","content": f"증상: {symptom}\n제품: {prod_model}\n근거문서: {titles}"}
    ]
    llm = await ollama.chat(model=model, messages=messages)
    plan = llm.get("message",{}).get("content","근거에 기반한 절차를 생성하지 못했습니다.")

    state["answer"] = plan
    state["citations"] = cites

    if confidence < 2 or len(cites) < 1:
        state.setdefault("actions", []).append({
            "type":"server_action","action":"CREATE_TICKET",
            "params":{"symptom": symptom, "product": state.get('product',{}), "priority":"high"}
        })
        state["answer"] += "\n\n(정확한 해결을 위해 담당자에게 보고를 진행합니다.)"
    return state

async def warranty_flow_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state.setdefault("actions", []).append({
        "type":"client_action","action":"REQUEST_INFO",
        "params":{"fields":["purchase_date","receipt_photo"],"tips":"영수증/구매내역 촬영 업로드"}
    })
    state["answer"] = "정품등록을 위해 구매일자와 영수증 이미지를 제출해주세요."
    state["citations"] = []
    return state

async def service_flow_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state.setdefault("actions", []).append({
        "type":"server_action","action":"CREATE_TICKET",
        "params":{"symptom": state.get('text',''), "product": state.get('product',{}), "priority":"normal"}
    })
    state["answer"] = "A/S 접수를 진행했어요. 접수 번호는 추후 안내됩니다."
    state["citations"] = []
    return state

async def action_suggester_node(state: Dict[str, Any]) -> Dict[str, Any]:
    text = (state.get("text") or "").lower()
    actions = state.get("actions") or []
    if "스캔" in text or "촬영" in text:
        actions.append({
            "type": "client_action",
            "action": "OPEN_CAMERA",
            "params": {"mode": "document", "timeoutMs": 15000},
            "safety": {"requires_user_confirm": True, "reason": "개인정보 포함 가능"}
        })
    state["actions"] = actions
    return state

```
### server/app/services/model_router.py
```python
from typing import Optional
from app.core.config import settings

LANG_MODEL_MAP = {
    "ko": "llama3:instruct",
    "en": "llama3:instruct",
}

def choose_model(lang: str | None, policy_country: Optional[str]) -> str:
    if lang:
        key = lang.split("-")[0]
        if key in LANG_MODEL_MAP:
            return LANG_MODEL_MAP[key]
    return settings.default_model

```
### server/app/services/ollama_client.py
```python
import httpx
from app.core.config import settings

class OllamaClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or settings.ollama_base_url
        self._client = httpx.AsyncClient(timeout=60)

    async def generate(self, model: str, prompt: str, **kwargs):
        url = f"{self.base_url}/api/generate"
        payload = {"model": model, "prompt": prompt, **kwargs}
        r = await self._client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

    async def chat(self, model: str, messages: list[dict], **kwargs):
        url = f"{self.base_url}/api/chat"
        payload = {"model": model, "messages": messages, **kwargs}
        r = await self._client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

ollama = OllamaClient()

```
### server/app/services/policy/enforcer.py
```python
from typing import Dict, List
from app.core.config import settings

SENSITIVE_TOKENS = ["주민등록번호", "여권번호"]

class PolicyEnforcer:
    def __init__(self, rules: Dict | None = None):
        self.rules = rules or {}

    def require_citations(self) -> bool:
        return bool(self.rules.get("require_citations", True))

    def mask_pii(self, text: str) -> str:
        masked = text
        for t in SENSITIVE_TOKENS:
            masked = masked.replace(t, "[민감정보]")
        return masked

    def filter_topics(self, text: str) -> str:
        prohibited = self.rules.get("prohibited_topics", [])
        for topic in prohibited:
            if topic in text:
                return "요청하신 내용은 정책상 제공할 수 없어요."
        return text

    def enforce(self, answer: str, citations: List[Dict]) -> tuple[str, List[Dict]]:
        out = self.mask_pii(answer) if settings.pii_masking == "strict" else answer
        out = self.filter_topics(out)
        if self.require_citations() and not citations:
            out += "\n\n(참고: 현재 답변에는 근거가 부족합니다.)"
        return out, citations

```
### server/app/services/policy/overlay_loader.py
```python
import yaml
from pathlib import Path
from typing import List, Dict

BASE = Path("graphs")

def load_overlays(ids: List[str]) -> Dict:
    result = {"rules": {}}
    for oid in ids:
        if oid.startswith("country:"):
            code = oid.split(":", 1)[1].lower()
            path = BASE / "overlays" / "country" / f"{code}.yaml"
        elif oid.startswith("policy:"):
            name = oid.split(":", 1)[1]
            path = BASE / "policies" / f"{name}.yaml"
        else:
            continue
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            result["rules"].update(data.get("rules", {}))
    return result

```
### server/app/services/rag/__init__.py
```python
from app.core.config import settings
from .service import RAGService as InMemRAG
try:
    from .service_qdrant import QdrantRAGService
    HAS_QDRANT = True
except Exception:
    HAS_QDRANT = False

if settings.qdrant_url and HAS_QDRANT:
    rag = QdrantRAGService()
else:
    rag = InMemRAG()

```
### server/app/services/rag/embeddings.py
```python
from sentence_transformers import SentenceTransformer
from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedder(model_name: str):
    return SentenceTransformer(model_name)

def embed(texts: list[str], model_name: str) -> list[list[float]]:
    em = get_embedder(model_name)
    vecs = em.encode(texts, normalize_embeddings=True).tolist()
    return vecs

```
### server/app/services/rag/indexer_qdrant.py
```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from app.core.config import settings
from .embeddings import embed

class QdrantIndexer:
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url) if settings.qdrant_url else None
        self.collection = settings.qdrant_collection

    def ensure_collection(self, dim: int):
        if not self.client:
            return
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection not in names:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert(self, docs: list[dict]):
        if not self.client:
            return {"ok": False}
        if not docs:
            return {"ok": True, "count": 0}
        texts = [d.get("content","") or "" for d in docs]
        vecs = embed(texts, settings.embed_model)
        dim = len(vecs[0])
        self.ensure_collection(dim)
        points = []
        for i, d in enumerate(docs):
            points.append(PointStruct(
                id=int(d.get("id") or i+1),
                vector=vecs[i],
                payload={
                    "title": d.get("title"),
                    "url": d.get("url"),
                    "date": d.get("date"),
                    "tags": d.get("tags", []),
                    "products": d.get("products", []),
                }
            ))
        self.client.upsert(collection_name=self.collection, points=points)
        return {"ok": True, "count": len(points)}

    def search(self, query: str, limit: int = 5, filters: dict | None = None):
        if not self.client:
            return []
        qv = embed([query], settings.embed_model)[0]
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        f = None
        if filters:
            conds = []
            if filters.get("tags"):
                conds.append(FieldCondition(key="tags", match=MatchAny(any=filters["tags"])))
            if filters.get("products"):
                conds.append(FieldCondition(key="products", match=MatchAny(any=filters["products"])))
            if conds:
                f = Filter(must=conds)
        res = self.client.search(collection_name=self.collection, query_vector=qv, limit=limit, query_filter=f)
        out = []
        for r in res:
            p = r.payload or {}
            out.append({
                "id": r.id, "score": r.score,
                "title": p.get("title"), "url": p.get("url"),
                "date": p.get("date"), "tags": p.get("tags", []),
                "products": p.get("products", []),
                "content": ""
            })
        return out

indexer = QdrantIndexer()

```
### server/app/services/rag/memory_store.py
```python
from typing import List, Dict, Optional
from datetime import datetime
import itertools

_id = itertools.count(1)

class InMemoryDocStore:
    def __init__(self):
        self.docs: List[Dict] = []

    def add(self, title: str, url: Optional[str], content: str, date: Optional[str] = None,
            tags: Optional[List[str]] = None, products: Optional[List[str]] = None) -> Dict:
        doc = {
            "id": str(next(_id)),
            "title": title,
            "url": url,
            "content": content,
            "date": date or datetime.now().date().isoformat(),
            "tags": tags or [],
            "products": products or []
        }
        self.docs.append(doc)
        return doc

    def upsert(self, doc: Dict) -> Dict:
        if not doc.get("id"):
            return self.add(doc.get("title","Untitled"), doc.get("url"), doc.get("content",""),
                            doc.get("date"), doc.get("tags"), doc.get("products"))
        for i, d in enumerate(self.docs):
            if d["id"] == doc["id"]:
                self.docs[i] = {**d, **doc}
                return self.docs[i]
        self.docs.append(doc)
        return doc

    def all(self) -> List[Dict]:
        return self.docs

docstore = InMemoryDocStore()

# seed
if not docstore.all():
    docstore.add("에어컨 설치 가이드(벽걸이)", "https://example.com/ac-install",
                 "실내기 브라켓 고정, 배수 경사 확인, 실외기 환기 공간 확보...",
                 tags=["guide","install"], products=["AC-1234"])
    docstore.add("에어컨 에러 E5 해결", "https://example.com/e5",
                 "E5는 통신 오류. 전원 재시작→실내외기 케이블 체결 확인→보드 점검 순서로 진행.",
                 tags=["troubleshooting","runbook"], products=["AC-1234"])

```
### server/app/services/rag/service.py
```python
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from app.core.config import settings
from .memory_store import docstore

def _keyword_score(q: str, text: str) -> int:
    s = 0
    for w in set(q.lower().split()):
        if len(w) >= 2:
            s += text.lower().count(w)
    return s

class RAGService:
    def __init__(self, min_citations: int = None, freshness_days: int = None):
        self.min_citations = min_citations or settings.rag_min_citations
        self.freshness_days = freshness_days or settings.rag_freshness_days

    def retrieve(self, query: str, products: List[str] | None = None, tags: List[str] | None = None) -> List[Dict]:
        q = query or ""
        hits: List[Tuple[int, Dict]] = []
        for d in docstore.all():
            score = _keyword_score(q, d["content"] + " " + d["title"])
            if products and d.get("products"):
                if any(p in d["products"] for p in products):
                    score += 3
            if tags and d.get("tags"):
                if any(t in d["tags"] for t in tags):
                    score += 2
            hits.append((score, d))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [h[1] for h in hits if h[0] > 0][: max(self.min_citations, 5)]

    def enforce_citations(self, answer: str, retrieved: List[Dict]) -> List[Dict]:
        recent_cut = datetime.now().date() - timedelta(days=self.freshness_days)
        cites = []
        for d in retrieved:
            date = d.get("date")
            if date:
                try:
                    if datetime.fromisoformat(date).date() < recent_cut:
                        continue
                except Exception:
                    pass
            cites.append({"title": d["title"], "url": d.get("url"), "date": d.get("date")})
        if len(cites) < self.min_citations:
            for d in retrieved:
                c = {"title": d["title"], "url": d.get("url"), "date": d.get("date")}
                if c not in cites:
                    cites.append(c)
                if len(cites) >= self.min_citations:
                    break
        return cites[: max(self.min_citations, 2)]

    def upsert(self, title: str, content: str, url: str | None, tags: List[str] | None, products: List[str] | None, date: str | None = None) -> Dict:
        return docstore.add(title, url, content, date=date, tags=tags, products=products)

```
### server/app/services/rag/service_qdrant.py
```python
from datetime import datetime, timedelta
from typing import List, Dict
from app.core.config import settings
from .indexer_qdrant import indexer

class QdrantRAGService:
    def __init__(self):
        self.min_citations = settings.rag_min_citations
        self.freshness_days = settings.rag_freshness_days

    def retrieve(self, query: str, products: List[str] | None = None, tags: List[str] | None = None) -> List[Dict]:
        filters = {"products": products, "tags": tags}
        return indexer.search(query=query, limit=max(self.min_citations, 5), filters=filters)

    def enforce_citations(self, answer: str, retrieved: List[Dict]) -> List[Dict]:
        recent_cut = datetime.now().date() - timedelta(days=self.freshness_days)
        cites = []
        for d in retrieved:
            date = d.get("date")
            if date:
                try:
                    if datetime.fromisoformat(date).date() < recent_cut:
                        continue
                except Exception:
                    pass
            cites.append({"title": d.get("title",""), "url": d.get("url"), "date": d.get("date")})
        if len(cites) < self.min_citations:
            for d in retrieved:
                c = {"title": d.get("title",""), "url": d.get("url"), "date": d.get("date")}
                if c not in cites: cites.append(c)
                if len(cites) >= self.min_citations: break
        return cites[: max(self.min_citations, 2)]

    def upsert(self, docs: List[Dict]):
        return indexer.upsert(docs)

rag_qdrant = QdrantRAGService()

```
### server/app/utils/pii.py
```python
import re

PII_PATTERNS = [
    re.compile(r"(\d{6}-?\d{7})")  # naive RRN pattern
]

def mask(text: str) -> str:
    out = text
    for p in PII_PATTERNS:
        out = p.sub("[민감정보]", out)
    return out

```
### server/docker-compose.yml
```yaml
version: "3.9"
services:
  api:
    build: .
    container_name: ai-api
    ports:
      - "8080:8080"
    env_file: .env
    depends_on:
      - ollama
      - qdrant
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama:/root/.ollama
    restart: unless-stopped
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant:/qdrant/storage
    restart: unless-stopped
volumes:
  ollama:
  qdrant:

```
### server/graphs/appliance.yaml
```yaml
graph:
  id: appliance-v1
  nodes:
    - id: intent_router
      type: router
    - id: require_product
      type: guard
    - id: install_flow
      type: flow
    - id: usage_flow
      type: flow
    - id: troubleshoot_flow
      type: flow
    - id: warranty_flow
      type: flow
    - id: service_flow
      type: flow
    - id: action_suggester
      type: action
  edges:
    - from: intent_router
      to: require_product
    - from: require_product
      to: install_flow
    - from: require_product
      to: usage_flow
    - from: require_product
      to: troubleshoot_flow
    - from: intent_router
      to: warranty_flow
    - from: intent_router
      to: service_flow
    - from: install_flow
      to: action_suggester
    - from: usage_flow
      to: action_suggester
    - from: troubleshoot_flow
      to: action_suggester
    - from: warranty_flow
      to: action_suggester
    - from: service_flow
      to: action_suggester
metadata:
  overlays: [country, policy]
  version: 1.0.0

```
### server/graphs/core.yaml
```yaml
graph:
  id: core-v1
  nodes:
    - id: intent_router
      type: router
    - id: rag_answer
      type: llm
    - id: llm_answer
      type: llm
    - id: action_suggester
      type: action
  edges:
    - from: intent_router
      to: rag_answer
    - from: intent_router
      to: llm_answer
    - from: rag_answer
      to: action_suggester
    - from: llm_answer
      to: action_suggester
metadata:
  overlays: [country, policy]
  version: 1.0.0

```
### server/graphs/overlays/country/kr.yaml
```yaml
rules:
  nudge_language: "ko-KR"
  prohibited_topics: ["주민등록번호_요구"]

```
### server/graphs/policies/std-privacy.yaml
```yaml
rules:
  require_citations: true
  camera_proactive_default: "opt-in"
  pii_masking: "strong"

```
### server/requirements.txt
```
fastapi>=0.111
uvicorn[standard]>=0.30
pydantic>=2.7
httpx>=0.27
pyyaml>=6.0
langgraph>=0.2.49
langchain-core>=0.3.0
numpy>=1.26
qdrant-client>=1.9.0
sentence-transformers>=2.6

```
### server/tests/test_smoke.py
```python
from fastapi.testclient import TestClient
from app.main import app

def test_healthz():
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_chat():
    c = TestClient(app)
    req = {
        "text": "영수증을 스캔해서 경비처리해줘",
        "lang": "ko-KR",
        "policy_context": {"country": "KR"}
    }
    r = c.post("/v1/chat", json=req)
    assert r.status_code == 200
    assert "answer" in r.json()

```
