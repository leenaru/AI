# Onboarding Agent — Monorepo Code v0.1 (OSI-only)

> 전체 동작 흐름을 갖춘 **실행 가능한 최소 스켈레톤**입니다. FastAPI + LangGraph 오케스트레이터(노드 샘플), GraphRAG(Parquet) + **FAISS** 하이브리드 검색과 **가중치 rerank**, Model Adapter Layer(vLLM/Ollama), Slack 기반 HITL 승인 워크플로, Valkey(Postgres) 기반 상태 저장, Helm/docker-compose 배포 스켈레톤을 포함합니다. **모든 라이선스는 OSI 승인**만 사용합니다.

---

## 📁 Repo 구조

```
onboarding-agent/
├─ LICENSE
├─ README.md
├─ docker-compose.yml
├─ configs/
│  ├─ app.yaml
│  └─ logging.yaml
├─ server/
│  ├─ app.py
│  ├─ core/
│  │  ├─ config.py
│  │  ├─ logging.py
│  │  └─ security.py
│  ├─ api/
│  │  ├─ routes.py
│  │  └─ deps.py
│  ├─ orchestrator/
│  │  ├─ graph.py
│  │  ├─ nodes/
│  │  │  ├─ guard.py
│  │  │  ├─ intent.py
│  │  │  ├─ plan.py
│  │  │  ├─ tools.py
│  │  │  ├─ synthesize.py
│  │  │  ├─ translate.py
│  │  │  └─ respond.py
│  │  └─ checkpointer/
│  │     └─ valkey_store.py
│  ├─ model_adapter/
│  │  ├─ base.py
│  │  ├─ router.py
│  │  └─ providers/
│  │     ├─ ollama.py
│  │     └─ vllm.py
│  ├─ rag/
│  │  ├─ pipeline.py
│  │  ├─ search.py
│  │  ├─ bm25.py
│  │  ├─ schemas.py
│  │  └─ utils.py
│  ├─ slack/
│  │  └─ routes.py
│  ├─ vision/
│  │  ├─ service.py
│  │  └─ storage.py
│  └─ ops/
│     ├─ metrics.py
│     └─ health.py
├─ scripts/
│  ├─ build_faiss.py
│  └─ reindex.py
├─ helm/
│  ├─ Chart.yaml
│  ├─ values.yaml
│  └─ templates/
│     ├─ deployment.yaml
│     ├─ service.yaml
│     ├─ ingress.yaml
│     └─ configmap.yaml
└─ mobile/
   ├─ android/OnboardingAgentGating.kt
   └─ ios/OnboardingAgentGating.swift
```

---

## 📄 LICENSE (Apache-2.0)

```text
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
... (표준 전문) ...
```

---

## 🧭 README.md

````md
# Onboarding Agent (OSI-only)

- Server: FastAPI + LangGraph(오케스트레이션), GraphRAG(Parquet) + FAISS, Model Adapter(vLLM/Ollama)
- Stores: Valkey(체크포인터/큐/레이트리밋), Postgres(영속/잡)
- HITL: Slack 승인 워크플로
- 배포: docker-compose, Helm(K8s)

## Quickstart
```bash
# 1) 로컬 스택
cp configs/app.yaml.example configs/app.yaml
docker compose up -d --build

# 2) 인덱스 빌드 (docs/ 샘플 넣고 실행)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # 예: fastapi uvicorn[standard] pydantic ... faiss-cpu
python scripts/build_faiss.py --config configs/app.yaml

# 3) 서버 실행
uvicorn server.app:app --reload
````

````

---

## ⚙️ configs/app.yaml (요약)
```yaml
server:
  host: 0.0.0.0
  port: 8000
  log_config: configs/logging.yaml

stores:
  valkey_url: "redis://valkey:6379/0"   # Valkey 호환
  postgres_dsn: "postgresql://postgres:postgres@postgres:5432/agent"
  s3:
    endpoint_url: "http://minio:9000"   # on-prem S3 호환
    bucket: "ai-vision-temp"
    access_key: "minio"
    secret_key: "minio123"
    secure: false

models:
  router: configs/model-router.yaml

rag:
  graph_store:
    type: parquet
    path: data/parquet/
  index:
    type: faiss
    factory: hnsw
    params:
      dim: 768
      hnsw: { M: 32, efConstruction: 200 }
  ingest:
    unit_summary: true
    incremental: true
  retrieval:
    k_dense: 40
    k_sparse: 40
    k_graph: 2
  merge:
    formula: "alpha*dense + beta*bm25 + gamma*graph + delta*recency"
    weights: { alpha: 0.55, beta: 0.25, gamma: 0.15, delta: 0.05 }
    cross_encoder: { enabled: true, top_m: 10, model: bge-reranker-v2-m3 }

local_gate:
  thresholds: { low: 0.45, high: 0.75 }
  require_tools: { troubleshooting: true, device_registration: true, smalltalk: false }
  redact: { pii: ["phone", "address", "email"] }
  clip: { max_chars: 800, keep_slots: true }
````

---

## 🐳 docker-compose.yml

```yaml
version: "3.9"
services:
  api:
    build: .
    command: uvicorn server.app:app --host 0.0.0.0 --port 8000
    ports: ["8000:8000"]
    environment:
      - APP_CONFIG=configs/app.yaml
    depends_on: [postgres, valkey, minio]

  postgres:
    image: postgres:16
    environment:
      - POSTGRES_PASSWORD=postgres
    ports: ["5432:5432"]
    volumes: [pgdata:/var/lib/postgresql/data]

  valkey:
    image: valkey/valkey:7
    ports: ["6379:6379"]

  minio:
    image: minio/minio:RELEASE.2025-01-05T00-00-00Z
    command: server /data --console-address ":9001"
    environment:
      - MINIO_ROOT_USER=minio
      - MINIO_ROOT_PASSWORD=minio123
    ports: ["9000:9000", "9001:9001"]
    volumes: [miniodata:/data]

  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]

  vllm:
    image: vllm/vllm-openai:latest
    ports: ["8001:8000"]

volumes:
  pgdata:
  miniodata:
```

---

## 🐍 server/app.py

```python
from fastapi import FastAPI
from server.core.config import settings
from server.api.routes import router as api_router
from server.ops.health import router as health_router
from server.slack.routes import router as slack_router

app = FastAPI(title="Onboarding Agent API")
app.include_router(health_router, prefix="/")
app.include_router(api_router, prefix="/")
app.include_router(slack_router, prefix="/")
```

---

## 🐍 server/core/config.py

```python
from pydantic_settings import BaseSettings
from pydantic import Field
import yaml, os

class Settings(BaseSettings):
    app_config: str = Field(default=os.getenv("APP_CONFIG", "configs/app.yaml"))
    cfg: dict = {}

    def load(self):
        with open(self.app_config, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        return self

settings = Settings().load()
```

---

## 🐍 server/api/routes.py

```python
from fastapi import APIRouter, UploadFile, File, Body
from typing import Any, Dict
from server.orchestrator.graph import run_conversation
from server.vision.service import analyze_image

router = APIRouter()

@router.post("/chat")
async def chat_endpoint(payload: Dict[str, Any]):
    return await run_conversation(payload)

@router.post("/vision/analyze")
async def vision_endpoint(file: UploadFile = File(...), tasks: str = Body("ocr,detect")):
    image_bytes = await file.read()
    return analyze_image(image_bytes, tasks.split(","))

@router.post("/actions/result")
async def actions_result(payload: Dict[str, Any]):
    # 클라이언트 액션 결과 수신 (예: 카메라 촬영 결과)
    return {"status": "ok"}
```

---

## 🐍 server/orchestrator/graph.py (LangGraph 스켈레톤)

```python
from typing import Dict, Any
from server.orchestrator.nodes.intent import detect_intent
from server.orchestrator.nodes.plan import plan_steps
from server.orchestrator.nodes.tools import execute_tools
from server.orchestrator.nodes.synthesize import synthesize
from server.orchestrator.nodes.translate import translate
from server.orchestrator.nodes.respond import respond
from server.ops.metrics import trace

async def run_conversation(payload: Dict[str, Any]):
    state: Dict[str, Any] = {"input": payload, "kb_hits": []}
    with trace("conversation"):
        intent = await detect_intent(state)
        state["intent"] = intent
        plan = await plan_steps(state)
        state["plan"] = plan
        exec_out = await execute_tools(state)
        state.update(exec_out)
        draft = await synthesize(state)
        state["draft"] = draft
        final = await translate(state)
        state["final"] = final
        return await respond(state)
```

---

## 🐍 server/orchestrator/nodes/intent.py

```python
from typing import Dict, Any

async def detect_intent(state: Dict[str, Any]):
    text = state["input"].get("text", "")
    # 간단 게이트: 키워드 + 점수(데모)
    score = 0.8 if any(k in text.lower() for k in ["등록", "추가", "사진", "에러", "오류"]) else 0.5
    label = "troubleshooting" if "오류" in text or "에러" in text else "faq"
    slots = {}
    return {"label": label, "score": score, "slots": slots}
```

---

## 🐍 server/orchestrator/nodes/plan.py

```python
from typing import Dict, Any, List

async def plan_steps(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    intent = state.get("intent", {})
    label = intent.get("label")
    plan = []
    if label == "troubleshooting":
        plan.append({"tool": "rag_search", "args": {"q": state["input"].get("text", "")}})
        if any(k in state["input"].get("text", "") for k in ["사진", "이미지", "첨부"]):
            plan.append({"tool": "request_image"})
    else:
        plan.append({"tool": "rag_search", "args": {"q": state["input"].get("text", "")}})
    return plan
```

---

## 🐍 server/orchestrator/nodes/tools.py

```python
from typing import Dict, Any
from server.rag.search import hybrid_search

async def execute_tools(state: Dict[str, Any]) -> Dict[str, Any]:
    plan = state.get("plan", [])
    kb_hits = []
    actions = []
    for step in plan:
        if step["tool"] == "rag_search":
            res = hybrid_search(step["args"]["q"])  # passages, scores, citations
            kb_hits.extend(res.get("passages", []))
        elif step["tool"] == "request_image":
            actions.append({"type": "open_camera"})
    return {"kb_hits": kb_hits, "actions": actions}
```

---

## 🐍 server/orchestrator/nodes/synthesize.py

```python
from typing import Dict, Any
from server.model_adapter.router import ModelRouter

router = ModelRouter.load_from_yaml("configs/model-router.yaml")

async def synthesize(state: Dict[str, Any]) -> str:
    prompt = f"사용자 질문: {state['input'].get('text','')}.\n근거:{[p['text'] for p in state.get('kb_hits',[])][:3]}\n정확하고 간결하게 답변하라."
    return router.generate(prompt)
```

---

## 🐍 server/orchestrator/nodes/translate.py

```python
from typing import Dict, Any

async def translate(state: Dict[str, Any]) -> str:
    # 데모: 그대로 반환 (다국어 매니저 연동 지점)
    return state.get("draft", "")
```

---

## 🐍 server/orchestrator/nodes/respond.py

```python
from typing import Dict, Any

async def respond(state: Dict[str, Any]):
    return {
        "messages": [state.get("final", "")],
        "actions": state.get("actions", []),
        "citations": [{"id": h.get("id"), "meta": h.get("meta")} for h in state.get("kb_hits", [])]
    }
```

---

## 🐍 server/orchestrator/checkpointer/valkey\_store.py

```python
import redis
from server.core.config import settings

_pool = redis.ConnectionPool.from_url(settings.cfg["stores"]["valkey_url"])  # Valkey 호환
r = redis.Redis(connection_pool=_pool)

def set_step(key: str, value: str, ttl_sec: int = 3600):
    r.set(key, value, ex=ttl_sec)

def get_step(key: str):
    v = r.get(key)
    return v.decode() if v else None
```

---

## 🐍 server/model\_adapter/base.py

```python
from abc import ABC, abstractmethod
from typing import Dict

class Provider(ABC):
    @abstractmethod
    def health(self) -> bool: ...
    @abstractmethod
    def generate(self, prompt: str, params: Dict = {}) -> str: ...
```

---

## 🐍 server/model\_adapter/providers/ollama.py

```python
import httpx
from server.model_adapter.base import Provider

class OllamaProvider(Provider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url, self.model = base_url, model

    def health(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, params: dict = {}) -> str:
        payload = {"model": self.model, "prompt": prompt, **params}
        r = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
        r.raise_for_status()
        # streaming 이벤트를 단순화하여 전체 텍스트만 합산
        text = "".join([line.get("response", "") for line in r.json().get("events", [])]) if r.headers.get("content-type"," ").startswith("application/json") else r.text
        return text
```

---

## 🐍 server/model\_adapter/providers/vllm.py

```python
import httpx
from server.model_adapter.base import Provider

class VLLMProvider(Provider):
    def __init__(self, base_url: str = "http://localhost:8001/v1", model: str = "qwen72b"):
        self.base_url, self.model = base_url, model

    def health(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/models", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, params: dict = {}) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        }
        payload.update(params)
        r = httpx.post(f"{self.base_url}/chat/completions", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
```

---

## 🐍 server/model\_adapter/router.py

```python
import yaml
from typing import Dict
from server.model_adapter.providers.ollama import OllamaProvider
from server.model_adapter.providers.vllm import VLLMProvider

class ModelRouter:
    def __init__(self, providers: Dict[str, object], prefer: list):
        self.providers = providers
        self.prefer = prefer

    @classmethod
    def load_from_yaml(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        providers = {
            "ollama": OllamaProvider(model=cfg["ollama"]["model"], base_url=cfg["ollama"]["base_url"]),
            "vllm": VLLMProvider(model=cfg["vllm"]["model"], base_url=cfg["vllm"]["base_url"]),
        }
        return cls(providers, cfg.get("prefer", ["ollama", "vllm"]))

    def generate(self, prompt: str) -> str:
        # 헬스체크 기반 선호도 순서로 시도, 실패 시 페일오버
        for key in self.prefer:
            prv = self.providers[key]
            if prv.health():
                try:
                    return prv.generate(prompt)
                except Exception:
                    continue
        # 모두 실패하면 마지막으로라도 시도
        return list(self.providers.values())[-1].generate(prompt)
```

---

## 🧾 configs/model-router.yaml

```yaml
ollama:
  base_url: http://ollama:11434
  model: llama3
vllm:
  base_url: http://vllm:8000/v1
  model: qwen72b
prefer: ["ollama", "vllm"]
```

---

## 🐍 server/rag/schemas.py

```python
from typing import List, Dict, Any
from pydantic import BaseModel

class Passage(BaseModel):
    id: str
    text: str
    meta: Dict[str, Any] = {}
    score: float = 0.0
```

---

## 🐍 server/rag/bm25.py

```python
from rank_bm25 import BM25Okapi
from typing import List
import re

def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

class BM25:
    def __init__(self, corpus: List[str]):
        self.tokens = [tokenize(t) for t in corpus]
        self.model = BM25Okapi(self.tokens)

    def search(self, q: str, k: int = 10):
        scores = self.model.get_scores(tokenize(q))
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return idx, [scores[i] for i in idx]
```

---

## 🐍 server/rag/pipeline.py (Graph→Parquet→Embed→FAISS)

```python
import os, json, uuid, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import faiss
from sentence_transformers import SentenceTransformer

PARQUET_DIR = "data/parquet"
FAISS_DIR = "data/faiss"
EMB_DIM = 768

_model = SentenceTransformer("intfloat/multilingual-e5-small")

# 문서 → 유닛/엣지 생성 (간이)
def graphify(docs):
    units, edges = [], []
    for doc_id, text in docs:
        chunks = [t.strip() for t in text.split("\n\n") if t.strip()]
        for c in chunks:
            uid = str(uuid.uuid4())
            units.append({"unit_id": uid, "doc_id": doc_id, "text": c, "created_at": int(time.time())})
            # 간단 엔티티/관계 샘플
            edges.append({"src": uid, "rel": "MENTIONS", "tgt": doc_id})
    return pd.DataFrame(units), pd.DataFrame(edges)

# 유닛 요약(데모: 원문 그대로)
def summarize_units(df_units: pd.DataFrame):
    df_units["summary"] = df_units["text"].str.slice(0, 200)
    return df_units

# 임베딩 + FAISS 인덱스 빌드
def build_faiss(df_units: pd.DataFrame):
    os.makedirs(FAISS_DIR, exist_ok=True)
    texts = df_units["summary"].tolist()
    embs = _model.encode(["query: "+t for t in texts], normalize_embeddings=True)
    index = faiss.IndexHNSWFlat(EMB_DIM, 32)
    index.hnsw.efConstruction = 200
    index.add(np.array(embs).astype("float32"))
    faiss.write_index(index, os.path.join(FAISS_DIR, "hnsw.index"))

# Parquet 저장
def save_parquet(df_units, df_edges):
    os.makedirs(PARQUET_DIR, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df_units), os.path.join(PARQUET_DIR, "units.parquet"))
    pq.write_table(pa.Table.from_pandas(df_edges), os.path.join(PARQUET_DIR, "edges.parquet"))

# 엔트리 포인트
def run_pipeline(docs):
    units, edges = graphify(docs)
    units = summarize_units(units)
    save_parquet(units, edges)
    build_faiss(units)
```

---

## 🐍 server/rag/search.py (GraphRAG+FAISS + weighted rerank)

```python
import os, numpy as np, faiss, pandas as pd, time
from typing import Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from .bm25 import BM25

PARQUET_DIR = "data/parquet"
FAISS_DIR = "data/faiss"
_emb = SentenceTransformer("intfloat/multilingual-e5-small")
_ce = None

cfg = {
  "k_dense": 40, "k_sparse": 40, "k_graph": 2,
  "weights": {"alpha":0.55,"beta":0.25,"gamma":0.15,"delta":0.05},
  "cross_encoder": {"enabled": True, "top_m": 10, "model": "BAAI/bge-reranker-v2-m3"}
}

def _load_units():
    df = pd.read_parquet(os.path.join(PARQUET_DIR, "units.parquet"))
    return df

def _faiss_index():
    return faiss.read_index(os.path.join(FAISS_DIR, "hnsw.index"))

_units_cache = None
_bm25 = None


def hybrid_search(q: str) -> Dict[str, Any]:
    global _units_cache, _bm25, _ce
    if _units_cache is None:
        _units_cache = _load_units()
        _bm25 = BM25(_units_cache["summary"].tolist())
    # DENSE
    qv = _emb.encode(["query: "+q], normalize_embeddings=True).astype("float32")
    index = _faiss_index()
    D, I = index.search(qv, cfg["k_dense"])  # distances
    dense_scores = 1 - (D[0])  # cosine approx

    # SPARSE(BM25)
    idx_bm25, bm25_scores = _bm25.search(q, k=cfg["k_sparse"]) if cfg["k_sparse"]>0 else ([],[])

    # GRAPH (Parquet 유닛 근접: 데모로 recency 사용)
    now = int(time.time())
    recency = (now - _units_cache.loc[I[0], "created_at"]).clip(lower=1)
    recency_scores = 1/np.log(recency+e := 2.71828)

    # Merge candidates
    candidates = {}
    def add(idx, s_dense=0, s_bm25=0, s_graph=0, s_rec=0):
        if idx not in candidates:
            candidates[idx] = {"dense":0,"bm25":0,"graph":0,"recency":0}
        candidates[idx]["dense"] = max(candidates[idx]["dense"], s_dense)
        candidates[idx]["bm25"] = max(candidates[idx]["bm25"], s_bm25)
        candidates[idx]["graph"] = max(candidates[idx]["graph"], s_graph)
        candidates[idx]["recency"] = max(candidates[idx]["recency"], s_rec)

    for rank, idx in enumerate(I[0]):
        add(int(idx), s_dense=float(dense_scores[rank]), s_rec=float(recency_scores[rank]))
    for rank, idx in enumerate(idx_bm25):
        add(int(idx), s_bm25=float(bm25_scores[rank]))

    w = cfg["weights"]
    for k,v in candidates.items():
        v["score"] = w["alpha"]*v["dense"] + w["beta"]*v["bm25"] + w["gamma"]*v["graph"] + w["delta"]*v["recency"]

    # top-N by weighted score
    top = sorted(candidates.items(), key=lambda kv: kv[1]["score"], reverse=True)[:cfg["cross_encoder"]["top_m"]]
    rows = [(i, _units_cache.iloc[i]["summary"]) for i,_ in top]

    # OPTIONAL cross-encoder rerank
    if cfg["cross_encoder"]["enabled"]:
        if _ce is None:
            _ce = CrossEncoder(cfg["cross_encoder"]["model"])  # loads on first use
        pairs = [(q, text) for _, text in rows]
        ces = _ce.predict(pairs)
        for idx, (i, _) in enumerate(rows):
            candidates[i]["score"] = float(ces[idx])
        top = sorted(candidates.items(), key=lambda kv: kv[1]["score"], reverse=True)[:10]

    passages = []
    for i,_ in top:
        row = _units_cache.iloc[i]
        passages.append({"id": str(row["unit_id"]), "text": row["summary"], "meta": {"doc_id": row["doc_id"]}, "score": candidates[i]["score"]})

    return {"passages": passages, "citations": [p["id"] for p in passages]}
```

---

## 🐍 server/vision/service.py

```python
from typing import List, Dict
from .storage import persist_temp_artifacts

def analyze_image(image_bytes: bytes, tasks: List[str]) -> Dict:
    # 데모: 바이트 길이와 요청 태스크 에코 + (옵션) 임시 보관
    ref = persist_temp_artifacts(image_bytes, meta={"tasks": tasks})
    return {"tasks": tasks, "bytes": len(image_bytes), "artifact_ref": ref}
```

---

## 🐍 server/vision/storage.py (S3/MinIO 임시 보관 - 선택)

```python
import boto3, os, hashlib, time
from server.core.config import settings

s3cfg = settings.cfg.get("stores", {}).get("s3", {})

s3 = boto3.client(
    "s3",
    endpoint_url=s3cfg.get("endpoint_url"),
    aws_access_key_id=s3cfg.get("access_key"),
    aws_secret_access_key=s3cfg.get("secret_key"),
    region_name="us-east-1",
    use_ssl=s3cfg.get("secure", False),
)

BUCKET = s3cfg.get("bucket", "ai-vision-temp")


def persist_temp_artifacts(image_bytes: bytes, meta: dict):
    if not image_bytes:
        return None
    h = hashlib.sha1(image_bytes).hexdigest()
    ts = int(time.time())
    key = f"temp/{ts}/{h}.bin"
    s3.put_object(Bucket=BUCKET, Key=key, Body=image_bytes)
    s3.put_object(Bucket=BUCKET, Key=key.replace(".bin", ".json"), Body=str(meta).encode())
    return {"s3": f"s3://{BUCKET}/{key}"}
```

---

## 🐍 server/slack/routes.py (HITL 승인 흐름)

```python
from fastapi import APIRouter, Request
from typing import Dict, Any
from server.rag.pipeline import run_pipeline

router = APIRouter()

@router.post("/kb/propose")
async def kb_propose(payload: Dict[str, Any]):
    # Slack 카드로 보낼 페이로드 생성 (데모)
    return {"ok": True, "message": "Proposed", "payload": payload}

@router.post("/kb/approve")
async def kb_approve(payload: Dict[str, Any]):
    # 승인 시: Parquet 파티션 증분 + FAISS add or rebuild
    docs = [(payload.get("doc_id", "ops"), payload.get("answer", ""))]
    run_pipeline(docs)  # 간이: 전체 파이프라인 재실행(데모)
    return {"ok": True, "message": "Indexed"}

@router.post("/kb/reject")
async def kb_reject(payload: Dict[str, Any]):
    return {"ok": True, "message": "Rejected", "reason": payload.get("reason")}
```

---

## 🐍 server/ops/metrics.py

```python
from contextlib import contextmanager
import time

@contextmanager
def trace(name: str):
    t0 = time.time()
    try:
        yield
    finally:
        dt = (time.time() - t0) * 1000
        print(f"[trace] {name} took {dt:.1f}ms")
```

---

## 🐍 server/ops/health.py

```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/healthz")
async def healthz():
    return {"status": "ok"}

@router.get("/readyz")
async def readyz():
    return {"status": "ready"}
```

---

## 🐍 scripts/build\_faiss.py

```python
import argparse, glob
from server.rag.pipeline import run_pipeline

def load_docs(path_glob: str):
    docs = []
    for p in glob.glob(path_glob):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            docs.append((p, f.read()))
    return docs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs/*.txt")
    ap.add_argument("--config", default="configs/app.yaml")
    args = ap.parse_args()
    docs = load_docs(args.docs)
    run_pipeline(docs)
    print("FAISS index built.")
```

---

## 🐍 scripts/reindex.py

```python
# 운영에서 부분 증분 갱신/리빌드 수행용(데모로 전체 재실행)
from server.rag.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline([("ops-update", "신규 트러블슈팅 답변 본문...")])
    print("Reindexed.")
```

---

## ☸️ helm/Chart.yaml

```yaml
apiVersion: v2
name: onboarding-agent
version: 0.1.0
appVersion: "0.1.0"
description: Onboarding Agent (OSI-only)
```

### ☸️ helm/values.yaml (요약)

```yaml
image:
  repository: ghcr.io/example/onboarding-agent
  tag: v0.1.0
service:
  type: ClusterIP
  port: 8000
env:
  APP_CONFIG: /config/app.yaml
volumes:
  - name: config
    mountPath: /config
```

### ☸️ helm/templates/deployment.yaml (요약)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: onboarding-agent }
spec:
  replicas: 2
  selector: { matchLabels: { app: onboarding-agent } }
  template:
    metadata: { labels: { app: onboarding-agent } }
    spec:
      containers:
        - name: api
          image: {{ .Values.image.repository }}:{{ .Values.image.tag }}
          ports: [{ containerPort: 8000 }]
          env:
            - name: APP_CONFIG
              value: {{ .Values.env.APP_CONFIG }}
          volumeMounts:
            - name: config
              mountPath: /config
      volumes:
        - name: config
          configMap: { name: onboarding-agent-config }
```

---

## 🤖 mobile/android/OnboardingAgentGating.kt (게이팅 스텁)

```kotlin
package ai.onboarding

object LocalGateCfg {
    const val LOW = 0.45
    const val HIGH = 0.75
}

class OnDeviceLLM {
    fun inferIntent(text: String): Triple<String, Double, Map<String, String>> {
        val label = if (text.contains("오류") || text.contains("에러")) "troubleshooting" else "faq"
        val score = if (text.contains("사진") || text.contains("이미지")) 0.7 else 0.5
        return Triple(label, score, emptyMap())
    }
}

class Gating {
    private val llm = OnDeviceLLM()
    fun route(text: String): Action {
        val (label, score, slots) = llm.inferIntent(text)
        return when {
            score < LocalGateCfg.LOW -> Action.Clarify("질문을 조금 더 구체적으로 말씀해 주세요")
            score < LocalGateCfg.HIGH -> Action.Escalate("/chat", mapOf("text" to text, "intent" to label, "slots" to slots))
            else -> Action.LocalAnswer("간단 답변: $label")
        }
    }
}

sealed class Action {
    data class Clarify(val msg: String): Action()
    data class Escalate(val path: String, val payload: Map<String, Any?>): Action()
    data class LocalAnswer(val msg: String): Action()
}
```

---

## 🍎 mobile/ios/OnboardingAgentGating.swift

```swift
import Foundation

struct LocalGateCfg { static let low = 0.45; static let high = 0.75 }

class OnDeviceLLM {
  func inferIntent(_ text: String) -> (String, Double, [String:String]) {
    let label = (text.contains("오류") || text.contains("에러")) ? "troubleshooting" : "faq"
    let score = (text.contains("사진") || text.contains("이미지")) ? 0.7 : 0.5
    return (label, score, [:])
  }
}

enum Action {
  case clarify(String)
  case escalate(String, [String:Any])
  case localAnswer(String)
}

class Gating {
  let llm = OnDeviceLLM()
  func route(_ text: String) -> Action {
    let (label, score, slots) = llm.inferIntent(text)
    if score < LocalGateCfg.low { return .clarify("질문을 조금 더 구체적으로 말씀해 주세요") }
    if score < LocalGateCfg.high { return .escalate("/chat", ["text": text, "intent": label, "slots": slots]) }
    return .localAnswer("간단 답변: \(label)")
  }
}
```

---

### ✅ 참고

* 모든 의존성은 OSI 승인 라이선스(예: FastAPI/MIT, Uvicorn/BSD, Pydantic/MIT, FAISS/MIT, NumPy/Pandas/BSD, PyArrow/Apache-2.0, rank\_bm25/MIT, sentence-transformers/Apache-2.0, slack\_sdk/MIT, boto3/Apache-2.0) 기반입니다.
* 실제 프로덕션에선 모델/토큰 한도, 프롬프트 템플릿, 정책 필터, 번역 매니저, 로깅/메트릭을 강화하세요.
* GraphDB는 **옵션**이며 Parquet로 시작 → 필요 시 JanusGraph/NebulaGraph로 확장.
