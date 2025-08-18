# LangGraph + Ollama 멀티턴 데모 (체크포인트 · 서브그래프 · 스트리밍 · 에러 복구 · 라우터/생성 이원화)

이 레포는 다음을 모두 포함합니다.

- **LangGraph** 멀티턴 예제들
- **체크포인트**: Memory / SQLite
- **서브그래프**: 수집/라우터/생성 분리
- **스트리밍**: values / updates (SSE)
- **에러 복구**: try/except 리커버리 · 재시도/폴백
- **이원화 템플릿**: 라우터 전용 소형 LLM + 생성 전용 대형 LLM

## 빠른 시작

```bash
# 1) Docker Compose로 Ollama + 앱 동시에 실행
cp .env.example .env  # 필요 시 모델 태그 수정
docker compose up --build
# 앱: http://localhost:8000  (FastAPI 문서: /docs)

# 2) 로컬 파이썬으로 실행(선택)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OLLAMA_HOST=http://localhost:11434
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> **모델 태그 주의**: 환경에 따라 `qwen3:8b-instruct`가 없을 수 있습니다. 그럴 경우 `qwen2.5:7b-instruct`/`qwen2.5:14b-instruct` 등 가용 instruct 태그로 교체하세요.

## 주요 환경변수(.env)

- `ROUTER_MODEL` (기본 `qwen2.5:3b-instruct`) : 경량 라우터용
- `GENERATOR_MODEL` (기본 `llama3:8b-instruct`) : 본문 생성용
- `DEFAULT_MODEL` (기본 `llama3:8b-instruct`)
- `OLLAMA_HOST` (컨테이너 내부 기본 `http://ollama:11434`)

## 엔드포인트 요약

- `GET  /health` : 상태 확인
- `POST /chat` : 단발 호출
  - JSON: `{ "prompt": "...", "graph": "llm_router_multiturn", "thread_id": "t1", "model": "..." }`
- `GET  /sse/values` : values 스트리밍
  - 쿼리: `?graph=collect_plan&prompt=...&thread_id=...`
- `GET  /sse/updates` : updates 스트리밍
  - 쿼리: `?graph=router_generate&prompt=...&thread_id=...`

## 그래프 이름

- `basic` : MemorySaver + 단일 노드
- `sqlite` : SqliteSaver + 단일 노드
- `collect_plan` : 수집(서브그래프) → 플랜
- `router_generate` : 라우터(서브그래프, LLM 판단) → 생성(서브그래프)
- `llm_router_multiturn` : LLM JSON 라우팅 + 멀티턴 슬롯필링 + **라우터/생성 모델 이원화**
- `recover_try` : try/except 리커버리 경로
- `retry_fallback` : 재시도 + 폴백 모델

## 모델 프리풀(선택)

```bash
./scripts/pull_models.sh
```

---

## 라이선스

MIT


---

## RAG 추가 (Basic RAG / HQ-RAG / GraphRAG)

- **임베딩 모델**: Ollama의 `nomic-embed-text`(권장) 또는 `mxbai-embed-large`
- **인덱스**: FAISS(`/data/index/faiss_basic`), HQ-RAG는 멀티쿼리+재랭킹(`/data/index/faiss_hq`), GraphRAG는 엔티티 그래프(`/data/index/graph.json`)
- 사전 준비:
  ```bash
  ollama pull nomic-embed-text
  # (옵션) ollama pull mxbai-embed-large
  ```

### 엔드포인트
- `POST /rag/basic` : 기본 RAG
- `POST /rag/hq`    : HQ-RAG (멀티쿼리+LLM 재랭킹)
- `POST /rag/graph` : GraphRAG (엔티티 그래프 기반)

### 인덱싱
- Basic/HQ 공통 인덱스 작성:
  ```bash
  python -m app.rag.index_basic --docs /data/docs --out /data/index/faiss_basic
  python -m app.rag.index_basic --docs /data/docs --out /data/index/faiss_hq   # 동일 인덱스 재사용 가능
  ```
- GraphRAG 그래프 구축:
  ```bash
  python -m app.rag.index_graph --docs /data/docs --out /data/index/graph.json
  ```


## Web UI (React) 데모

```
cd webui
# Node 18+ 권장
npm install
npm run dev   # http://localhost:5173 (백엔드: http://localhost:8000)
```
- 상단 드롭다운에서 **RAG 모드(Basic/HQ/Graph)** 선택
- **New Chat**을 누르면 새로운 `thread_id`로 멀티턴 세션 시작
- 메시지 전송 시 백엔드의 `/rag/chat/{mode}`(멀티턴 RAG) 엔드포인트를 호출


## SSE 실시간 스트리밍 UI

- 백엔드: `GET /sse/rag?mode=basic|hybrid&q=...&thread_id=...` (토큰 단위 전송: `event: token`)
- 프런트: `webui/src/App.tsx`의 `Stream SSE` 체크박스 활성화 → 실시간 표시

## 하이브리드 검색 / BM25

- 인덱싱:
  ```bash
  python -m app.rag.hybrid.index_hybrid --docs /data/docs --faiss_out /data/index/faiss_hybrid --bm25_out /data/index/bm25.json
  ```
- 실행:
  ```bash
  curl -X POST http://localhost:8000/rag/hybrid -H 'Content-Type: application/json' -d '{"query":"시드니 가족 일정", "k":6}'
  ```

## 문서 업로드 & 온라인 인덱싱

- 업로드: `POST /admin/upload` (multipart/form-data, 필드명 `file`)  
- 재인덱싱: `POST /admin/reindex` (기본/고품질/하이브리드 인덱스 모두 갱신)

프런트엔드에서 `.txt/.md` 파일 선택 → 자동 업로드 후 재인덱싱 수행.

## Redis / Postgres 체크포인터

- 환경변수:
  - `CHKPT_BACKEND=memory|sqlite|redis|postgres`
  - `REDIS_URL=redis://localhost:6379/0` (옵션) `REDIS_NAMESPACE=lg`
  - `POSTGRES_URL=postgresql://user:pass@localhost:5432/langgraph`
- 그래프: `checkpointers` 선택 시 해당 백엔드로 상태 영속화.

> 주의: `langgraph.checkpoint.redis` 및 `langgraph.checkpoint.postgres`는 설치/버전 환경에 따라 가용성이 다릅니다. `requirements.txt`에 `redis`, `psycopg2-binary`가 추가되어 있으니, 서버 환경에 맞춰 URL을 설정하세요.


## WebSocket 실시간 스트리밍

- 엔드포인트: `ws://localhost:8000/ws/rag?mode=basic|hybrid`
- 메시지 전송: 텍스트로 질문을 전송하면 서버가 순서대로 `ctx`(컨텍스트/출처), `token`(토큰 델타들), `done` 이벤트를 JSON으로 보냅니다.
- 프런트: `webui/src/App.tsx` 에서 **Use WS** 체크박스 활성화

## 하이라이트/출처 표기 UI

- Sources 패널에 점수/출처와 함께 스니펫을 표시하고, **질문 키워드 하이라이트**(`<mark>`)

## BM25+Vector 하이브리드 튜너

- `POST /admin/hybrid/tune` → `/data/index/hybrid_config.json`에 저장
- 파라미터: `weight_vec`, `weight_bm25`, `fuse: max|sum|rrf`, `rrf_k`
- 런타임 검색은 구성값을 읽어 병합 방식과 가중치를 조정

## 문서 삭제/부분 인덱싱 (Delta)

- `POST /admin/delta/add_text` : 텍스트를 즉시 BM25/FAISS 하이브리드 인덱스에 추가 (FAISS는 add_texts)
- `POST /admin/delta/delete` : 특정 `source`를 삭제 대상으로 마킹하고 BM25 JSON에서 즉시 제거 (FAISS는 tombstone 후 `POST /admin/compact`로 전체 재색인)
- `POST /admin/compact` : tombstone 목록을 반영하여 /data/docs + /data/uploads 기반으로 **완전 재색인**
