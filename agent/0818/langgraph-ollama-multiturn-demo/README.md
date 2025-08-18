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
