# Server Skeleton – LangGraph + Ollama (v0.1)

This document contains a runnable **server skeleton** for your system: FastAPI + LangGraph runtime + Ollama model router + HQ‑RAG stub + Policy overlays. It’s designed to be **model/locale/policy pluggable** and production‑oriented (env‑driven, structured logs, tests, Docker).

---

## Repo Tree

```
server/
├─ README.md
├─ requirements.txt
├─ docker-compose.yml
├─ Dockerfile
├─ .env.example
├─ app/
│  ├─ main.py
│  ├─ core/
│  │  ├─ config.py
│  │  └─ logging.py
│  ├─ models/
│  │  └─ schemas.py
│  ├─ api/
│  │  └─ v1/
│  │     ├─ chat.py
│  │     └─ events.py
│  ├─ services/
│  │  ├─ ollama_client.py
│  │  ├─ model_router.py
│  │  ├─ langgraph/
│  │  │  ├─ graph_runner.py
│  │  │  └─ nodes.py
│  │  ├─ rag/
│  │  │  ├─ service.py
│  │  │  └─ memory_store.py
│  │  └─ policy/
│  │     ├─ overlay_loader.py
│  │     └─ enforcer.py
│  └─ utils/
│     └─ pii.py
├─ graphs/
│  ├─ core.yaml
│  ├─ overlays/
│  │  └─ country/
│  │     └─ kr.yaml
│  └─ policies/
│     └─ std-privacy.yaml
└─ tests/
   └─ test_smoke.py
```

---

## README.md

````md
# Server – LangGraph + Ollama (Skeleton)

## Quickstart

### 1) Python
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
````

### 2) Docker (with Ollama)

```bash
docker compose up -d
# API: http://localhost:8080
docker compose logs -f api
```

**Tip:** ensure Ollama has at least one model pulled, e.g.:

```bash
docker exec -it ollama ollama pull llama3:instruct
```

## Endpoints (v1)

- `POST /v1/chat` – main multimodal chat/intent endpoint
- `POST /v1/events/camera` – ingest camera proactive events

## Config

Copy `.env.example` → `.env` and adjust.

## Tests

```bash
pytest -q
```

````

---

## requirements.txt
```txt
fastapi>=0.111
uvicorn[standard]>=0.30
pydantic>=2.7
httpx>=0.27
pyyaml>=6.0
langgraph>=0.2.49
langchain-core>=0.3.0
numpy>=1.26
````

---

## .env.example

```env
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

# Graph
GRAPH_FILE=graphs/core.yaml
```

---

## app/core/config.py

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

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## app/core/logging.py

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

---

## app/models/schemas.py

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class DeviceContext(BaseModel):
    platform: Optional[str]
    network: Optional[str]
    battery: Optional[float]
    permissions: Optional[Dict[str, bool]]

class PolicyContext(BaseModel):
    country: Optional[str]
    age_mode: Optional[str] = "adult"

class MultimodalPayload(BaseModel):
    image_refs: Optional[List[str]] = None
    audio_ref: Optional[str] = None

class ChatRequest(BaseModel):
    type: str = Field("graph_query")
    lang: str = Field("ko-KR")
    text: Optional[str] = None
    channels: List[str] = ["text"]
    multimodal: Optional[MultimodalPayload] = None
    device_context: Optional[DeviceContext] = None
    policy_context: Optional[PolicyContext] = None
    overlays: Optional[List[str]] = None

class Citation(BaseModel):
    title: str
    url: str
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
```

---

## app/services/ollama\_client.py

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

---

## app/services/model\_router.py

```python
from typing import Optional
from app.core.config import settings

# Simple policy: route by language prefix; fallback to DEFAULT_MODEL
LANG_MODEL_MAP = {
    "ko": "llama3:instruct",   # replace if you have a better ko-optimized model
    "en": "llama3:instruct",
}

def choose_model(lang: str | None, policy_country: Optional[str]) -> str:
    if lang:
        key = lang.split("-")[0]
        if key in LANG_MODEL_MAP:
            return LANG_MODEL_MAP[key]
    return settings.default_model
```

---

## app/services/rag/memory\_store.py

```python
from typing import List, Dict
from datetime import datetime

class InMemoryDocStore:
    def __init__(self):
        self.docs: List[Dict] = []

    def add(self, title: str, url: str, content: str, date: str | None = None):
        self.docs.append({"title": title, "url": url, "content": content, "date": date})

    def all(self) -> List[Dict]:
        return self.docs

docstore = InMemoryDocStore()
# Seed example docs
if not docstore.all():
    docstore.add("샘플 A", "https://example.com/a", "이것은 예시 콘텐츠입니다.", date=datetime.now().date().isoformat())
    docstore.add("샘플 B", "https://example.com/b", "추가 예시 콘텐츠입니다.")
```

---

## app/services/rag/service.py

```python
from typing import List, Dict
from datetime import datetime, timedelta
from app.core.config import settings
from .memory_store import docstore

class RAGService:
    def __init__(self, min_citations: int = None, freshness_days: int = None):
        self.min_citations = min_citations or settings.rag_min_citations
        self.freshness_days = freshness_days or settings.rag_freshness_days

    def retrieve(self, query: str) -> List[Dict]:
        # naive retrieval: return top-N by keyword presence
        q = query.lower()
        hits = []
        for d in docstore.all():
            score = (d["content"].lower().count(q) + d["title"].lower().count(q))
            hits.append((score, d))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [h[1] for h in hits[: max(self.min_citations, 3)]]

    def enforce_citations(self, answer: str, retrieved: List[Dict]) -> List[Dict]:
        # filter by freshness first
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
            cites.append({"title": d["title"], "url": d["url"], "date": d.get("date")})
        # ensure at least min citations; if not, fallback to include older
        if len(cites) < self.min_citations:
            for d in retrieved:
                c = {"title": d["title"], "url": d["url"], "date": d.get("date")}
                if c not in cites:
                    cites.append(c)
                if len(cites) >= self.min_citations:
                    break
        return cites[: max(self.min_citations, 2)]

rag = RAGService()
```

---

## app/services/policy/overlay\_loader.py

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
            data = yaml.safe_load(path.read_text()) or {}
            result["rules"].update(data.get("rules", {}))
    return result
```

---

## app/services/policy/enforcer.py

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
        # PII masking → topic filter → citation presence
        out = self.mask_pii(answer) if settings.pii_masking == "strict" else answer
        out = self.filter_topics(out)
        if self.require_citations() and not citations:
            out += "\n\n(참고: 현재 답변에는 근거가 부족합니다.)"
        return out, citations

enforcer = PolicyEnforcer()
```

---

## app/services/langgraph/nodes.py

```python
from typing import Dict, Any
from app.services.rag.service import rag
from app.services.model_router import choose_model
from app.services.ollama_client import ollama

async def intent_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # very simple: route to rag if query seems like a fact ask, else to llm
    text = state.get("text") or ""
    if any(k in text for k in ["무엇", "언제", "어디", "정의", "설명", "근거"]):
        state["route"] = "rag_answer"
    else:
        state["route"] = "llm_answer"
    return state

async def rag_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("text") or ""
    retrieved = rag.retrieve(query)
    cites = rag.enforce_citations("", retrieved)
    # synthesize a short answer via model (optional)
    model = choose_model(state.get("lang"), state.get("policy_country"))
    messages = [{"role": "user", "content": f"질문: {query}\n참고자료 제목: {[d['title'] for d in retrieved]}\n간결히 요약 답변."}]
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

async def action_suggester_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # Example: suggest client camera open if keywords present
    text = (state.get("text") or "").lower()
    actions = []
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

---

## app/services/langgraph/graph\_runner.py

```python
from typing import Dict, Any
import yaml
from app.core.config import settings
from app.services.policy.overlay_loader import load_overlays
from app.services.policy.enforcer import PolicyEnforcer
from .nodes import intent_router_node, rag_answer_node, llm_answer_node, action_suggester_node

# Optional: use LangGraph if available; otherwise run a simple dispatcher
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
        sg.add_node("action_suggester", action_suggester_node)
        sg.set_entry_point("intent_router")
        # static edges; runtime will read `route` field
        sg.add_edge("intent_router", "rag_answer")
        sg.add_edge("intent_router", "llm_answer")
        sg.add_edge("rag_answer", "action_suggester")
        sg.add_edge("llm_answer", "action_suggester")
        self.app = sg.compile()

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # augment state with overlay context
        if state.get("policy_context", {}).get("country"):
            state["policy_country"] = state["policy_context"]["country"]
        state["lang"] = state.get("lang") or "ko-KR"

        if LANGGRAPH_AVAILABLE:
            # Execute graph until terminal node
            result = await self.app.ainvoke(state)
        else:
            # Fallback: manual routing
            result = await intent_router_node(state)
            if result.get("route") == "rag_answer":
                result = await rag_answer_node(result)
            else:
                result = await llm_answer_node(result)
            result = await action_suggester_node(result)

        # policy enforcement
        ans, cites = self.enforcer.enforce(result.get("answer", ""), result.get("citations", []))
        result["answer"], result["citations"] = ans, cites
        return result

def get_runner() -> GraphRunner:
    return GraphRunner(graph_file=settings.graph_file, overlays=settings.policy_overlays)
```

---

## app/api/v1/chat.py

```python
from fastapi import APIRouter, Depends
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

---

## app/api/v1/events.py

```python
from fastapi import APIRouter
from app.models.schemas import CameraEvent, Answer

router = APIRouter()

@router.post("/events/camera", response_model=Answer)
async def camera_event(ev: CameraEvent):
    # naive policy: if a high-risk event is detected, ask client to confirm opening camera
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

---

## app/utils/pii.py

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

---

## app/main.py

```python
from fastapi import FastAPI
from app.core.logging import setup_logging
from app.core.config import settings
from app.api.v1 import chat, events

setup_logging("INFO")
app = FastAPI(title="On-device + Server AI – API", version="0.1.0")

app.include_router(chat.router, prefix="/v1")
app.include_router(events.router, prefix="/v1")

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "env": settings.app_env}
```

---

## graphs/core.yaml

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

---

## graphs/overlays/country/kr.yaml

```yaml
rules:
  nudge_language: "ko-KR"
  prohibited_topics: ["주민등록번호_요구"]
```

---

## graphs/policies/std-privacy.yaml

```yaml
rules:
  require_citations: true
  camera_proactive_default: "opt-in"
  pii_masking: "strong"
```

---

## docker-compose.yml

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
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama:/root/.ollama
    restart: unless-stopped
volumes:
  ollama:
```

---

## Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## tests/test\_smoke.py

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
    # skeleton: answer string exists
    assert "answer" in r.json()
```



---

# Appliance Scenario Extensions (Install → Register → Warranty → Usage → Troubleshoot → A/S)

이 섹션은 가전 설치/등록/정품등록/사용법/고장 대처/AS 요청까지 **엔드투엔드 시나리오**를 서버 골격에 추가하는 코드와 그래프 예시를 포함합니다. 핵심은 **HQ‑RAG 근거 기반**과 **트러블슈팅 정확도 우선**, **미해결 시 담당자 에스컬레이션 → RAG 업데이트** 자동 루프입니다.

## 0) 새/변경 파일 트리

```
server/
└─ app/
   ├─ models/schemas.py                # (+) ProductInfo, Ticket, RAG Ingest
   ├─ api/v1/
   │  ├─ chat.py                       # (same)
   │  ├─ events.py                     # (same)
   │  ├─ tickets.py                    # (+) 에스컬레이션/해결 반영
   │  └─ rag_admin.py                  # (+) RAG 수동/담당자 인입
   ├─ services/
   │  ├─ rag/
   │  │  ├─ service.py                 # (~) retrieve 고도화, upsert 지원
   │  │  ├─ memory_store.py            # (~) id/업데이트/간단 쿼리
   │  │  └─ indexer.py                 # (+) (stub) 향후 벡터DB로 교체 지점
   │  └─ langgraph/
   │     ├─ nodes.py                   # (~) 시나리오 노드/트러블슈팅 노드 추가
   │     └─ graph_runner.py            # (same)
└─ graphs/
   └─ appliance.yaml                   # (+) 가전 시나리오 그래프
```

---

## 1) models/schemas.py (추가)

```python
# ... (기존 import 유지)
class ProductInfo(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    serial: Optional[str] = None

class TroubleshootContext(BaseModel):
    symptom: Optional[str] = None
    severity: Optional[str] = None  # info|warn|error|critical

class RAGDocIngest(BaseModel):
    title: str
    url: Optional[str] = None
    content: str
    date: Optional[str] = None
    tags: List[str] = []            # ["guide","troubleshooting","verified","warranty"] 등
    products: List[str] = []        # ["AC-1234","WM-900N"]

class Ticket(BaseModel):
    id: Optional[str] = None
    user_id: str
    product: Optional[ProductInfo] = None
    symptom: str
    status: str = "open"            # open|pending|resolved
    notes: List[str] = []

class TicketResolve(BaseModel):
    resolution: str
    add_to_rag: bool = True
    tags: List[str] = ["troubleshooting","verified"]
    url: Optional[str] = None
    products: List[str] = []
```

---

## 2) services/rag/memory\_store.py (업데이트)

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

---

## 3) services/rag/service.py (업데이트)

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
        # ensure min
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

rag = RAGService()
```

---

## 4) services/rag/indexer.py (신규)

```python
# Placeholder for future vector index (Qdrant/pgvector). For now, no-op.
class Indexer:
    def reindex(self):
        return {"status": "ok", "indexed": "inmemory"}

indexer = Indexer()
```

---

## 5) api/v1/rag\_admin.py (신규)

```python
from fastapi import APIRouter
from app.models.schemas import RAGDocIngest
from app.services.rag.service import rag
from app.services.rag.indexer import indexer

router = APIRouter()

@router.post("/rag/ingest")
async def rag_ingest(doc: RAGDocIngest):
    saved = rag.upsert(title=doc.title, content=doc.content, url=doc.url, tags=doc.tags, products=doc.products, date=doc.date)
    return {"ok": True, "doc": saved}

@router.post("/rag/reindex")
async def rag_reindex():
    return indexer.reindex()
```

---

## 6) api/v1/tickets.py (신규)

```python
from fastapi import APIRouter, HTTPException
from typing import Dict
from app.models.schemas import Ticket, TicketResolve
from app.services.rag.service import rag

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
    if body.add_to_rag:
        rag.upsert(title=f"해결: {t.symptom}", content=body.resolution, url=body.url, tags=body.tags, products=[t.product.model] if (t.product and t.product.model) else body.products)
    return {"ok": True, "ticket": t}
```

---

## 7) services/langgraph/nodes.py (추가 노드)

```python
# ... 기존 import 유지
from app.models.schemas import ProductInfo

async def intent_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    text = state.get("text") or ""
    x = text.lower()
    if any(k in x for k in ["설치", "연결", "추가", "페어링"]):
        state["route"] = "install_flow"
    elif any(k in x for k in ["등록", "정품"]):
        state["route"] = "warranty_flow"
    elif any(k in x for k in ["사용법", "어떻게", "기능"]):
        state["route"] = "usage_flow"
    elif any(k in x for k in ["고장", "에러", "오류", "안됨", "소음", "누수"]):
        state["route"] = "troubleshoot_flow"
    elif any(k in x for k in ["as", "수리", "방문", "접수"]):
        state["route"] = "service_flow"
    elif any(k in x for k in ["무엇", "언제", "어디", "정의", "설명", "근거"]):
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

async def install_flow_node(state: Dict[str, Any]) -> Dict[str, Any]:
    q = f"설치 가이드 {state.get('product',{}).get('model','')}"
    retrieved = rag.retrieve(q, products=[state.get('product',{}).get('model','')], tags=["guide","install"])
    cites = rag.enforce_citations("", retrieved)
    model = choose_model(state.get("lang"), state.get("policy_country"))
    messages = [{"role":"user","content": f"다음 자료 기반으로 설치 절차를 단계별 bullet로 간결히 요약: {[d['title'] for d in retrieved]}"}]
    llm = await ollama.chat(model=model, messages=messages)
    state["answer"] = llm.get("message",{}).get("content","설치 절차를 찾지 못했습니다.")
    state["citations"] = cites
    return state

async def usage_flow_node(state: Dict[str, Any]) -> Dict[str, Any]:
    q = f"사용법 {state.get('text','')} {state.get('product',{}).get('model','')}"
    retrieved = rag.retrieve(q, products=[state.get('product',{}).get('model','')], tags=["guide","faq"])
    cites = rag.enforce_citations("", retrieved)
    model = choose_model(state.get("lang"), state.get("policy_country"))
    messages = [{"role":"user","content": f"아래 문서를 근거로 사용법을 요약: {[d['title'] for d in retrieved]}"}]
    llm = await ollama.chat(model=model, messages=messages)
    state["answer"] = llm.get("message",{}).get("content","도움을 찾지 못했습니다.")
    state["citations"] = cites
    return state

async def troubleshoot_flow_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # 정확도 우선: 증상/모델/환경 수집 → 근거 기반 단계 실행
    prod_model = state.get('product',{}).get('model')
    symptom = (state.get('troubleshoot',{}) or {}).get('symptom') or state.get('text','')
    # 필요 시 사진 요청
    if not prod_model:
        state.setdefault("actions", []).append({
            "type":"client_action","action":"OPEN_CAMERA","params":{"mode":"label"},
            "safety":{"requires_user_confirm": True, "reason":"모델 라벨 인식"}
        })
    retrieved = rag.retrieve(symptom, products=[prod_model] if prod_model else None, tags=["troubleshooting","runbook"])
    cites = rag.enforce_citations("", retrieved)

    # 간이 confidence: verified 태그 포함 + 제품 매칭 유무
    confidence = 0
    if any("verified" in d.get("tags",[]) for d in retrieved): confidence += 1
    if prod_model and any(prod_model in d.get("products",[]) for d in retrieved): confidence += 1
    if len(retrieved) >= 2: confidence += 1

    model = choose_model(state.get("lang"), state.get("policy_country"))
    prompt = (
        "당신은 가전 A/S 기술 문서 기반의 수리 가이드입니다.
"
        "응답 형식:
"
        "1) 원인 후보(우선순위)
2) 안전 경고
3) 단계별 조치(체크리스트)
4) 검증 방법
"
        "반드시 근거 문서의 범위를 벗어나지 말고, 모호하면 '확신 부족'으로 표시하세요."
    )
    titles = [d['title'] for d in retrieved]
    messages = [
        {"role":"system","content": prompt},
        {"role":"user","content": f"증상: {symptom}
제품: {prod_model}
근거문서: {titles}"}
    ]
    llm = await ollama.chat(model=model, messages=messages)
    plan = llm.get("message",{}).get("content","근거에 기반한 절차를 생성하지 못했습니다.")

    state["answer"] = plan
    state["citations"] = cites

    # 에스컬레이션 조건: confidence < 2 또는 근거 부족
    if confidence < 2 or len(cites) < 1:
        state.setdefault("actions", []).append({
            "type":"server_action","action":"CREATE_TICKET",
            "params":{"symptom": symptom, "product": state.get('product',{}), "priority":"high"}
        })
        state["answer"] += "

(정확한 해결을 위해 담당자에게 보고를 진행합니다.)"
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
```

---

## 8) graphs/appliance.yaml (신규 그래프)

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

---

## 9) app/api/main.py 라우팅 추가 (발췌)

```python
# app/main.py
from app.api.v1 import chat, events
from app.api.v1 import rag_admin, tickets

app.include_router(chat.router, prefix="/v1")
app.include_router(events.router, prefix="/v1")
app.include_router(rag_admin.router, prefix="/v1")
app.include_router(tickets.router, prefix="/v1")
```

---

## 10) 예시 호출(트러블슈팅 → 에스컬레이션 → 담당자 해결 등록 → RAG 업데이트)

### 1) 사용자가 고장 증상 문의

```bash
curl -s http://localhost:8080/v1/chat -H 'Content-Type: application/json' -d '{
  "text": "에어컨이 E5 에러가 떠요",
  "lang": "ko-KR",
  "product": {"brand":"Acme","model":"AC-1234"},
  "policy_context": {"country":"KR"}
}'
```

- 응답: 근거 기반 단계 + (confidence 낮을 경우) `actions[].server_action=CREATE_TICKET`

### 2) 서버/오케스트레이터가 티켓 생성

```bash
curl -s http://localhost:8080/v1/tickets -H 'Content-Type: application/json' -d '{
  "user_id":"u-100",
  "product":{"brand":"Acme","model":"AC-1234"},
  "symptom":"E5 에러로 냉방 불가"
}'
```

### 3) 담당자 해결 입력(해결과정/근거 링크 포함, RAG 업데이트)

```bash
curl -s -X PATCH http://localhost:8080/v1/tickets/1/resolve \
  -H 'Content-Type: application/json' -d '{
    "resolution": "E5는 실내외기 통신. 실외기 커넥터 CN-Comm 재결속으로 해결됨. 케이블 단선 확인 체크리스트 포함.",
    "url":"https://kb.example.com/ac/e5-comm",
    "tags":["troubleshooting","verified"],
    "products":["AC-1234"]
  }'
```

- 결과: `rag.upsert(...)`로 RAG에 즉시 반영 → 다음 유사 질문에 근거로 사용됨.

---

## 11) 운용 메모

- **정확도 우선 모드**: `troubleshoot_flow`는 근거 밖 추론을 금지하고 confidence 낮으면 즉시 에스컬레이션.
- **사진/추가정보 요청**: `require_product_context` 노드와 `OPEN_CAMERA`, `REQUEST_INFO` 액션으로 구현.
- **RAG 교체**: `services/rag/indexer.py` 지점을 Qdrant/pgvector로 대체하고 `retrieve`에 임베딩 검색 추가 예정.
- **안전성**: 전기/배관 등 위험 단계는 항상 "안전 경고" 섹션을 선두에 배치하도록 프롬프트 고정.



---

# 12) 클라이언트 액션 루프 샘플 (Android & iOS)

서버 응답의 `actions[]`를 클라이언트가 안전하게 실행하고 결과를 서버로 업링크하는 **오케스트레이션 루프** 예시입니다.

## 12.1 공통 액션 계약(서버→클라)

```json
{
  "actions": [
    {
      "type": "client_action",
      "action": "OPEN_CAMERA",
      "params": {"mode": "document", "timeoutMs": 15000},
      "safety": {"requires_user_confirm": true, "reason": "개인정보 포함 가능"}
    },
    {
      "type": "client_action",
      "action": "REQUEST_INFO",
      "params": {"fields": ["brand","model","serial"]}
    },
    {
      "type": "server_action",
      "action": "CREATE_TICKET",
      "params": {"symptom":"E5 에러", "priority":"high"}
    }
  ]
}
```

### 결과 업링크(클라→서버)

- 카메라: `/v1/events/camera` (예: `{user_id, event_type:"document_captured", meta:{image_ref:"..."}}`)
- 정보입력: `/v1/chat` 재호출 시 `product`/추가 필드 포함
- 서버 액션은 클라가 직접 호출(`/v1/tickets`)하거나, 앱 오케스트레이터가 서버로 위임

---

## 12.2 Android (Kotlin) 샘플

```kotlin
// build.gradle: CameraX / ActivityResult API 사용 권장
// implementation("androidx.activity:activity-ktx:1.9.0")
// implementation("androidx.camera:camera-camera2:1.3.4")
// implementation("androidx.camera:camera-lifecycle:1.3.4")
// implementation("androidx.camera:camera-view:1.3.4")

sealed class ClientAction {
    data class OpenCamera(val mode: String, val timeoutMs: Long?): ClientAction()
    data class RequestInfo(val fields: List<String>): ClientAction()
}

data class ServerAction(val name: String, val params: Map<String, Any?>)

data class ActionEnvelope(
    val clientActions: List<ClientAction> = emptyList(),
    val serverActions: List<ServerAction> = emptyList()
)

fun parseActions(json: String): ActionEnvelope {
    // pseudo: Gson/Moshi로 파싱해서 매핑
    // ... 생략 (프로덕션에서는 sealed adapter 사용)
    return ActionEnvelope()
}

class ActionOrchestrator(
    private val ctx: Context,
    private val api: ApiClient,
) {
    suspend fun handle(envelope: ActionEnvelope) {
        // 1) 서버 액션은 즉시 호출
        envelope.serverActions.forEach { sa ->
            when (sa.name) {
                "CREATE_TICKET" -> api.createTicket(sa.params)
                else -> {/*noop*/}
            }
        }
        // 2) 클라이언트 액션 실행
        envelope.clientActions.forEach { ca ->
            when (ca) {
                is ClientAction.OpenCamera -> openCamera(ca)
                is ClientAction.RequestInfo -> requestInfo(ca)
            }
        }
    }

    private suspend fun openCamera(a: ClientAction.OpenCamera) {
        // 권한 체크 → 사용자 확인(privacy reason 표시) → CameraX 프리뷰/캡처 → 파일 저장
        val ok = ensureCameraPermission(ctx)
        if (!ok) return
        val imageUri = captureWithCameraX(ctx) // 구현부 생략 (PreviewView + ImageCapture)
        // 업링크
        api.postCameraEvent(
            userId = api.userId,
            eventType = "document_captured",
            meta = mapOf("image_ref" to imageUri.toString(), "mode" to a.mode)
        )
    }

    private suspend fun requestInfo(a: ClientAction.RequestInfo) {
        val data = showFormDialog(ctx, fields = a.fields) // 사용자 입력 UI
        // /v1/chat 재호출(제품 정보 포함)
        api.chat(
            text = "정보 업데이트",
            product = data, // {brand, model, serial}
        )
    }
}
```

> 참고: 샘플에서는 `captureWithCameraX`, `showFormDialog`, `ApiClient` 구현을 생략했습니다. 실제 앱에서는 **Activity Result API**로 카메라/갤러리 결괄를 비동기 수신하고, 결과를 서버에 업링크하세요.

---

## 12.3 iOS (SwiftUI) 샘플

```swift
import SwiftUI
import AVFoundation
import PhotosUI

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
    @Published var requestedFields: [String] = []

    func handle(_ env: ActionEnvelope) {
        // 서버 액션 먼저
        for sa in env.serverActions {
            if sa.name == "CREATE_TICKET" { callCreateTicket(sa.params) }
        }
        // 클라이언트 액션
        for ca in env.clientActions {
            switch ca {
            case .openCamera(let mode, _):
                requestCameraPermission { granted in
                    if granted { DispatchQueue.main.async { self.showCamera = true } }
                }
            case .requestInfo(let fields):
                DispatchQueue.main.async { self.requestedFields = fields }
            }
        }
    }

    private func requestCameraPermission(completion: @escaping (Bool)->Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { completion($0) }
        default: completion(false)
        }
    }

    private func callCreateTicket(_ params: [String:Any]) {
        // URLSession으로 /v1/tickets 호출
    }
}

struct CameraSheet: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        return picker
    }
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
}

struct ContentView: View {
    @StateObject var orchestrator = ActionOrchestrator()
    var body: some View {
        VStack {
            Button("서버 응답 처리") {
                // 데모용 가짜 액션
                let env = ActionEnvelope(
                    clientActions: [.openCamera(mode: "document", timeoutMs: 15000), .requestInfo(fields: ["brand","model"])],
                    serverActions: [("CREATE_TICKET", ["symptom":"소음"])])
                orchestrator.handle(env)
            }
        }
        .sheet(isPresented: $orchestrator.showCamera) { CameraSheet() }
    }
}
```

---

# 13) Qdrant 연동 (RAG 실전화)

인메모리 RAG를 **Qdrant 벡터DB + 멀티링구얼 임베딩**으로 교체하는 스켈레톤입니다.

## 13.1 의존성/환경

- `requirements.txt` 추가

```txt
qdrant-client>=1.9.0
sentence-transformers>=2.6
```

- `.env.example` 추가

```env
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=kb_docs
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

- `docker-compose.yml`에 Qdran
