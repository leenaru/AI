# 온보딩 에이전트 — v0.4 패치 (다국어·능력 라우팅 + Model Adapter Layer)
> 생성 시각: 2025-08-10T17:00:12.845136Z

이 패치는 **다국어·능력 기반 라우팅**을 `Model Adapter Layer` 위에서 수행하도록 확장합니다.

- `server/config/model_profiles.yaml` : 백엔드/모델별 언어·능력·비용·컨텍스트 정의
- `server/config/router_policy.yaml` : 하드 룰/가중치/폴백 정책
- `app/services/model_router.py` : 점수 기반 선택(`choose_backend_and_model`)
- `app/services/adapters/registry.py` : `get_adapter(backend_id)` 지원
- `app/services/langgraph/nodes.py` : 각 노드가 동적으로 어댑터/모델 선택

## 실행
```bash
cd server
cp .env.example .env
docker compose up -d
curl -s http://localhost:8080/healthz
```

## 예시 호출
```bash
curl -s http://localhost:8080/v1/chat -H 'Content-Type: application/json' -d '{
  "text":"에어컨이 E5 에러가 떠요",
  "lang":"ko-KR",
  "product":{"brand":"Acme","model":"AC-1234"},
  "policy_context":{"country":"KR"}
}'
```

---

## 포함 파일 목록

- `server/.env.example`
- `server/Dockerfile`
- `server/app/core/config.py`
- `server/app/core/logging.py`
- `server/app/main.py`
- `server/app/services/adapters/ollama_adapter.py`
- `server/app/services/adapters/openai_adapter.py`
- `server/app/services/adapters/registry.py`
- `server/app/services/adapters/tgi_adapter.py`
- `server/app/services/adapters/vllm_adapter.py`
- `server/app/services/langgraph/nodes.py`
- `server/app/services/model_router.py`
- `server/app/services/rag/memory_store.py`
- `server/app/services/rag/service.py`
- `server/config/model_profiles.yaml`
- `server/config/router_policy.yaml`
- `server/docker-compose.yml`
- `server/requirements.txt`
- `server/tests/test_smoke.py`

---
### server/.env.example
```
APP_ENV=dev
API_PORT=8080

LLM_BACKEND=vllm

VLLM_BASE_URL=http://vllm:8000/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3

OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_DEFAULT_MODEL=llama3:instruct

OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-REPLACE
OPENAI_MODEL=gpt-4o

TGI_BASE_URL=http://tgi:8080
TGI_MODEL=meta-llama/Meta-Llama-3-8B-Instruct

RAG_MIN_CITATIONS=2
RAG_FRESHNESS_DAYS=60
POLICY_OVERLAYS=country:KR,policy:std-privacy
PII_MASKING=strict

GRAPH_FILE=graphs/appliance.yaml

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
### server/app/core/config.py
```python
from pydantic import BaseSettings, Field
from typing import List
import os

class Settings(BaseSettings):
    app_env: str = Field(default=os.getenv("APP_ENV", "dev"))
    api_port: int = Field(default=int(os.getenv("API_PORT", 8080)))
    llm_backend: str = Field(default=os.getenv("LLM_BACKEND", "vllm"))
    vllm_base_url: str = Field(default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    vllm_api_key: str = Field(default=os.getenv("VLLM_API_KEY", "EMPTY"))
    vllm_model: str = Field(default=os.getenv("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"))
    ollama_base_url: str = Field(default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_default_model: str = Field(default=os.getenv("OLLAMA_DEFAULT_MODEL", "llama3:instruct"))
    openai_base_url: str = Field(default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = Field(default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    tgi_base_url: str = Field(default=os.getenv("TGI_BASE_URL", "http://localhost:8080"))
    tgi_model: str = Field(default=os.getenv("TGI_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"))
    rag_min_citations: int = Field(default=int(os.getenv("RAG_MIN_CITATIONS", 2)))
    rag_freshness_days: int = Field(default=int(os.getenv("RAG_FRESHNESS_DAYS", 60)))
    policy_overlays: List[str] = Field(default_factory=lambda: os.getenv("POLICY_OVERLAYS", "").split(",") if os.getenv("POLICY_OVERLAYS") else [])
    pii_masking: str = Field(default=os.getenv("PII_MASKING", "strict"))
    graph_file: str = Field(default=os.getenv("GRAPH_FILE", "graphs/core.yaml"))
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
        payload = {"ts": time.time(), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        if record.exc_info: payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)
def setup_logging(level="INFO"):
    h = logging.StreamHandler(sys.stdout); h.setFormatter(JsonFormatter())
    root = logging.getLogger(); root.handlers.clear(); root.addHandler(h); root.setLevel(level)

```
### server/app/main.py
```python
from fastapi import FastAPI
from app.core.logging import setup_logging
from app.core.config import settings

setup_logging("INFO")
app = FastAPI(title="On-device + Server AI – API", version="0.4.1")

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "env": settings.app_env, "backend": settings.llm_backend}

```
### server/app/services/adapters/ollama_adapter.py
```python
import httpx
from typing import List, Dict, Any
from app.core.config import settings

class OllamaAdapter:
    def __init__(self, base: str | None = None):
        self.base = base or settings.ollama_base_url
    async def chat(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        payload = {"model": model or settings.ollama_default_model, "messages": messages}
        async with httpx.AsyncClient(timeout=60) as cx:
            r = await cx.post(f"{self.base}/api/chat", json=payload); r.raise_for_status()
            data = r.json()
            content = (data.get("message") or {}).get("content") or data.get("response") or ""
            return {"message": {"content": content}}

```
### server/app/services/adapters/openai_adapter.py
```python
import httpx
from typing import List, Dict, Any
from app.core.config import settings

class OpenAIAdapter:
    def __init__(self, base: str | None = None, api_key: str | None = None):
        self.base = (base or settings.openai_base_url).rstrip("/")
        self.key  = api_key or settings.openai_api_key
    async def chat(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=60) as cx:
            r = await cx.post(f"{self.base}/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"},
                json={"model": model or settings.openai_model, "messages": messages, "stream": False})
            r.raise_for_status(); data = r.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return {"message": {"content": content}}

```
### server/app/services/adapters/registry.py
```python
from typing import Dict
import yaml
from pathlib import Path
from app.services.adapters.ollama_adapter import OllamaAdapter
from app.services.adapters.vllm_adapter import VLLMAdapter
from app.services.adapters.openai_adapter import OpenAIAdapter
from app.services.adapters.tgi_adapter import TGIAdapter

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"  # server/config

def _load_profiles() -> dict:
    with open(CONFIG_DIR / "model_profiles.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

_PROFILES = _load_profiles()
_CACHE: Dict[str, object] = {}

def get_adapter(backend_id: str | None = None):
    if backend_id is None:
        backend_id = (_PROFILES.get("backends") or [{}])[0].get("id", "ollama_pool")

    if backend_id in _CACHE:
        return _CACHE[backend_id]

    be = next((b for b in _PROFILES.get("backends", []) if b["id"] == backend_id), None)
    if not be:
        be = next((b for b in _PROFILES.get("backends", []) if b["id"] == "ollama_pool"), None)

    t = (be or {}).get("type", "ollama")
    base_url = (be or {}).get("base_url")
    api_key = (be or {}).get("api_key")

    if t == "vllm":
        adapter = VLLMAdapter(base=base_url, api_key=api_key)
    elif t == "openai":
        adapter = OpenAIAdapter(base=base_url, api_key=api_key)
    elif t == "tgi":
        adapter = TGIAdapter(base=base_url)
    else:
        adapter = OllamaAdapter(base=base_url)

    _CACHE[backend_id] = adapter
    return adapter

llm_adapter = get_adapter()

```
### server/app/services/adapters/tgi_adapter.py
```python
import httpx
from typing import List, Dict, Any
from app.core.config import settings

class TGIAdapter:
    def __init__(self, base: str | None = None):
        self.base = (base or settings.tgi_base_url).rstrip("/")
    async def chat(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        prompt = "\n".join([m.get("content","") for m in messages])
        async with httpx.AsyncClient(timeout=60) as cx:
            r = await cx.post(f"{self.base}/generate", json={"inputs": prompt, "parameters": {}})
            r.raise_for_status(); data = r.json()
            text = data.get("generated_text") or (data[0].get("generated_text") if isinstance(data, list) else "")
            return {"message": {"content": text or ""}}

```
### server/app/services/adapters/vllm_adapter.py
```python
import httpx
from typing import List, Dict, Any
from app.core.config import settings

class VLLMAdapter:
    def __init__(self, base: str | None = None, api_key: str | None = None):
        self.base = (base or settings.vllm_base_url).rstrip("/")
        self.key  = api_key or settings.vllm_api_key
    async def chat(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=60) as cx:
            r = await cx.post(f"{self.base}/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"},
                json={"model": model or settings.vllm_model, "messages": messages, "stream": False})
            r.raise_for_status(); data = r.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return {"message": {"content": content}}

```
### server/app/services/langgraph/nodes.py
```python
from typing import Dict, Any
from app.services.rag.service import rag
from app.services.model_router import choose_backend_and_model
from app.services.adapters.registry import get_adapter

def _select_adapter_and_model(state: Dict[str, Any]):
    req_info = {
        "lang": state.get("lang") or "en",
        "capabilities": state.get("capabilities") or ["general"],
        "context_need": len((state.get("text") or "")) + 2048,
        "task": state.get("task") or "general",
        "multimodal": "image" if (state.get("multimodal") and state["multimodal"].get("image_refs")) else None,
        "quality": state.get("quality") or "normal",
    }
    sel = choose_backend_and_model(req_info)
    adapter = get_adapter(sel["backend"])
    return adapter, sel["model"]

async def intent_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    text = (state.get("text") or "").lower()
    if any(k in text for k in ["설치", "연결", "추가", "페어링"]): state["route"] = "install_flow"
    elif any(k in text for k in ["등록", "정품"]): state["route"] = "warranty_flow"
    elif any(k in text for k in ["사용법", "어떻게", "기능"]): state["route"] = "usage_flow"
    elif any(k in text for k in ["고장", "에러", "오류", "안됨", "소음", "누수"]): state["route"] = "troubleshoot_flow"
    elif any(k in text for k in ["as", "수리", "방문", "접수"]): state["route"] = "service_flow"
    elif any(k in text for k in ["무엇", "언제", "어디", "정의", "설명", "근거"]): state["route"] = "rag_answer"
    else: state["route"] = "llm_answer"
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
        state["answer"] = "제품 모델명/시리얼을 알려주십시오. 필요 시 제품 라벨 촬영을 도와드리겠습니다."
    return state

async def rag_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("text") or ""
    retrieved = rag.retrieve(query)
    cites = rag.enforce_citations("", retrieved)
    adapter, model = _select_adapter_and_model(state)
    titles = [d.get("title") for d in retrieved]
    messages = [{"role":"user","content": f"질문: {query}
참고자료 제목: {titles}
간결히 요약 답변."}]
    llm = await adapter.chat(model=model, messages=messages)
    state["answer"] = (llm.get("message") or {}).get("content", "요약을 생성하지 못했습니다.")
    state["citations"] = cites
    return state

async def llm_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    adapter, model = _select_adapter_and_model(state)
    messages = [{"role":"user","content": state.get("text") or ""}]
    llm = await adapter.chat(model=model, messages=messages)
    state["answer"] = (llm.get("message") or {}).get("content", "응답을 생성하지 못했습니다.")
    state["citations"] = []
    return state

async def install_flow_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prod_model = (state.get('product') or {}).get('model', '')
    q = f"설치 가이드 {prod_model}"
    retrieved = rag.retrieve(q, products=[prod_model] if prod_model else None, tags=["guide","install"])
    cites = rag.enforce_citations("", retrieved)
    adapter, model = _select_adapter_and_model(state)
    messages = [{"role":"user","content": f"다음 자료 기반으로 설치 절차를 단계별 bullet로 간결히 요약: {[d.get('title') for d in retrieved]}"}]
    llm = await adapter.chat(model=model, messages=messages)
    state["answer"] = (llm.get("message") or {}).get("content","설치 절차를 찾지 못했습니다.")
    state["citations"] = cites
    return state

async def usage_flow_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prod_model = (state.get('product') or {}).get('model', '')
    q = f"사용법 {state.get('text','')} {prod_model}"
    retrieved = rag.retrieve(q, products=[prod_model] if prod_model else None, tags=["guide","faq"])
    cites = rag.enforce_citations("", retrieved)
    adapter, model = _select_adapter_and_model(state)
    messages = [{"role":"user","content": f"아래 문서를 근거로 사용법을 요약: {[d.get('title') for d in retrieved]}"}]
    llm = await adapter.chat(model=model, messages=messages)
    state["answer"] = (llm.get("message") or {}).get("content","도움을 찾지 못했습니다.")
    state["citations"] = cites
    return state

async def troubleshoot_flow_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prod_model = (state.get('product') or {}).get('model')
    symptom = (state.get('troubleshoot') or {}).get('symptom') or state.get('text','')
    if not prod_model:
        state.setdefault("actions", []).append({"type":"client_action","action":"OPEN_CAMERA","params":{"mode":"label"},"safety":{"requires_user_confirm": True, "reason":"모델 라벨 인식"}})
    retrieved = rag.retrieve(symptom, products=[prod_model] if prod_model else None, tags=["troubleshooting","runbook"])
    cites = rag.enforce_citations("", retrieved)
    confidence = 0
    if any("verified" in (d.get("tags") or []) for d in retrieved): confidence += 1
    if prod_model and any(prod_model in (d.get("products") or []) for d in retrieved): confidence += 1
    if len(retrieved) >= 2: confidence += 1

    adapter, model = _select_adapter_and_model(state)
    prompt = (
        "당신은 가전 A/S 기술 문서 기반의 수리 가이드입니다.
"
        "응답 형식:
1) 원인 후보(우선순위)
2) 안전 경고
3) 단계별 조치(체크리스트)
4) 검증 방법
"
        "반드시 근거 문서의 범위를 벗어나지 말고, 모호하면 '확신 부족'으로 표시하세요."
    )
    titles = [d.get('title') for d in retrieved]
    messages = [
        {"role":"system","content": prompt},
        {"role":"user","content": f"증상: {symptom}
제품: {prod_model}
근거문서: {titles}"}
    ]
    llm = await adapter.chat(model=model, messages=messages)
    plan = (llm.get("message") or {}).get("content","근거에 기반한 절차를 생성하지 못했습니다.")

    state["answer"] = plan
    state["citations"] = cites

    if confidence < 2 or len(cites) < 1:
        state.setdefault("actions", []).append({"type":"server_action","action":"CREATE_TICKET","params":{"symptom": symptom, "product": state.get('product',{}), "priority":"high"}})
        state["answer"] += "\n\n(정확한 해결을 위해 담당자에게 보고를 진행합니다.)"
    return state

async def warranty_flow_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state.setdefault("actions", []).append({"type":"client_action","action":"REQUEST_INFO","params":{"fields":["purchase_date","receipt_photo"],"tips":"영수증/구매내역 촬영 업로드"}})
    state["answer"] = "정품등록을 위해 구매일자와 영수증 이미지를 제출해 주십시오."
    state["citations"] = []
    return state

async def service_flow_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state.setdefault("actions", []).append({"type":"server_action","action":"CREATE_TICKET","params":{"symptom": state.get('text',''), "product": state.get('product',{}), "priority":"normal"}})
    state["answer"] = "A/S 접수를 진행했습니다. 접수 번호는 추후 안내됩니다."
    state["citations"] = []
    return state

```
### server/app/services/model_router.py
```python
import math
import yaml
from typing import Dict, Any, Tuple
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"  # server/config

def _load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

_PROFILES = _load_yaml("model_profiles.yaml")
_POLICY = _load_yaml("router_policy.yaml")

def _score_model(req: Dict[str, Any], model: Dict[str, Any], weights: Dict[str, float]) -> float:
    s = 0.0
    req_lang = (req.get("lang") or "en").split("-")[0]
    langs = model.get("langs", [])
    if "*" in langs or req_lang in langs:
        s += weights.get("lang_match", 0.0)
    need_caps = set(req.get("capabilities") or [])
    has_caps = set(model.get("capabilities") or [])
    s += weights.get("capability_match", 0.0) * len(need_caps & has_caps)
    need_ctx = int(req.get("context_need", 4096))
    if int(model.get("max_context", 8192)) >= need_ctx:
        s += weights.get("max_context_need", 0.0)
    cost = float(model.get("cost", 1.0))
    s -= weights.get("cost_penalty", 0.0) * max(cost - 1.0, 0.0)
    return s

def _match_hard_rules(req: Dict[str, Any]) -> Tuple[str, str] | None:
    for rule in _POLICY.get("routing", {}).get("hard_rules", []):
        cond = rule.get("when", {})
        if all(req.get(k) == v for k, v in cond.items()):
            backend_id = rule["prefer"]
            be = next((b for b in _PROFILES.get("backends", []) if b["id"] == backend_id), None)
            if be and be.get("models"):
                return backend_id, be["models"][0]["name"]
    return None

def choose_backend_and_model(req: Dict[str, Any]) -> Dict[str, str]:
    hard = _match_hard_rules(req)
    if hard:
        return {"backend": hard[0], "model": hard[1]}

    weights = _POLICY.get("routing", {}).get("soft_weights", {})
    candidates = []
    for be in _PROFILES.get("backends", []):
        for m in be.get("models", []):
            candidates.append((be["id"], m["name"], _score_model(req, m, weights)))
    candidates.sort(key=lambda x: x[2], reverse=True)

    if candidates:
        be_id, model_name, _ = candidates[0]
        return {"backend": be_id, "model": model_name}

    default_be = _POLICY.get("routing", {}).get("default_backend")
    be = next((b for b in _PROFILES.get("backends", []) if b["id"] == default_be), None)
    if be and be.get("models"):
        return {"backend": default_be, "model": be["models"][0]["name"]}
    return {"backend": "ollama_pool", "model": "llama3:instruct"}

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

    def all(self) -> List[Dict]: return self.docs

docstore = InMemoryDocStore()
if not docstore.all():
    docstore.add("에어컨 설치 가이드(벽걸이)", "https://example.com/ac-install",
                 "실내기 브라켓 고정, 배수 경사 확인, 실외기 환기 공간 확보...", tags=["guide","install"], products=["AC-1234"])
    docstore.add("에어컨 에러 E5 해결", "https://example.com/e5",
                 "E5는 통신 오류. 전원 재시작→실내외기 케이블 체결 확인→보드 점검 순서.", tags=["troubleshooting","runbook"], products=["AC-1234"])

```
### server/app/services/rag/service.py
```python
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from .memory_store import docstore

def _keyword_score(q: str, text: str) -> int:
    s = 0
    for w in set(q.lower().split()):
        if len(w) >= 2: s += text.lower().count(w)
    return s

class RAGService:
    def __init__(self, min_citations: int = 2, freshness_days: int = 60):
        self.min_citations = min_citations
        self.freshness_days = freshness_days

    def retrieve(self, query: str, products: List[str] | None = None, tags: List[str] | None = None) -> List[Dict]:
        q = query or ""; hits: List[Tuple[int, Dict]] = []
        for d in docstore.all():
            score = _keyword_score(q, d["content"] + " " + d["title"])
            if products and d.get("products") and any(p in d["products"] for p in products): score += 3
            if tags and d.get("tags") and any(t in d["tags"] for t in tags): score += 2
            hits.append((score, d))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [h[1] for h in hits if h[0] > 0][:5]

    def enforce_citations(self, answer: str, retrieved: List[Dict]) -> List[Dict]:
        recent_cut = datetime.now().date() - timedelta(days=self.freshness_days)
        cites = []
        for d in retrieved:
            date = d.get("date")
            if date:
                try:
                    if datetime.fromisoformat(date).date() < recent_cut: continue
                except Exception: pass
            cites.append({"title": d["title"], "url": d.get("url"), "date": d.get("date")})
        if len(cites) < self.min_citations:
            for d in retrieved:
                c = {"title": d["title"], "url": d.get("url"), "date": d.get("date")}
                if c not in cites: cites.append(c)
                if len(cites) >= self.min_citations: break
        return cites[: max(self.min_citations, 2)]

rag = RAGService()

```
### server/config/model_profiles.yaml
```yaml
backends:
  - id: vllm_main
    type: vllm
    base_url: http://vllm:8000/v1
    api_key: EMPTY
    models:
      - name: mistralai/Mistral-7B-Instruct-v0.3
        langs: [en, fr, es, de]
        capabilities: [general, rag, function_call]
        max_context: 32768
        cost: 1.0
      - name: meta-llama/Meta-Llama-3-8B-Instruct
        langs: [en, ko, ja]
        capabilities: [general, rag]
        cost: 1.2

  - id: ollama_pool
    type: ollama
    base_url: http://ollama:11434
    models:
      - name: qwen2.5:7b-instruct
        langs: [zh, en, ko, ja]
        capabilities: [general, translate, rag]
        cost: 0.6
      - name: llama3:instruct
        langs: [en, ko]
        capabilities: [general, rag]
        cost: 0.5

  - id: gpt4o_ext
    type: openai
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    models:
      - name: gpt-4o
        langs: [*]
        capabilities: [general, vision, translate, code, math]
        cost: 5.0

```
### server/config/router_policy.yaml
```yaml
routing:
  default_backend: vllm_main
  hard_rules:
    - when: { multimodal: image }
      prefer: gpt4o_ext
    - when: { task: translate, quality: high }
      prefer: gpt4o_ext
  soft_weights:
    lang_match: 3.0
    capability_match: 2.0
    cost_penalty: 1.0
    latency_penalty: 1.0
    max_context_need: 1.5
  fallbacks:
    - ollama_pool
    - vllm_main
    - gpt4o_ext

```
### server/docker-compose.yml
```yaml
version: "3.9"
services:
  api:
    build: .
    container_name: ai-api
    ports: ["8080:8080"]
    env_file: .env
    depends_on:
      - vllm
      - ollama
      - qdrant

  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm
    command: >
      --model ${VLLM_MODEL}
      --port 8000
      --tensor-parallel-size 1
    ports: ["8000:8000"]
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports: ["11434:11434"]
    volumes: [ "ollama:/root/.ollama" ]
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports: ["6333:6333"]
    volumes: [ "qdrant:/qdrant/storage" ]
    restart: unless-stopped

volumes:
  ollama:
  qdrant:

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
def test_placeholder():
    assert 1 == 1

```
