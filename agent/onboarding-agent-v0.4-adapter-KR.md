# 온보딩 에이전트 — v0.4 (Model Adapter Layer)
이 문서는 **Model Adapter Layer**를 적용한 서버 골격의 전체 소스와 한글 설명을 하나로 묶은 파일입니다.

- 백엔드 교체( **vLLM / Ollama / OpenAI 호환 / HuggingFace TGI** ) 시에도 **LangGraph·서비스 코드 수정 없이** 동작
- `app/services/adapters/`에 어댑터 구현(통일 인터페이스: `adapter.chat(model, messages)`)
- HQ‑RAG(인메모리/선택적 Qdrant), 가전 시나리오 그래프, 티켓·RAG 인입 API 포함

## 빠른 시작 (Docker)
```bash
cd server
cp .env.example .env

# 백엔드 선택 (예: vLLM)
# LLM_BACKEND=vllm
# VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3

docker compose up -d
curl -s http://localhost:8080/healthz

# 테스트 호출
curl -s http://localhost:8080/v1/chat -H 'Content-Type: application/json' -d '{
  "text": "에어컨이 E5 에러가 떠요",
  "lang": "ko-KR",
  "product": {"brand":"Acme","model":"AC-1234"},
  "policy_context": {"country":"KR"}
}'
```

## 핵심 구조
- `adapters/base.py`: 어댑터 표준 인터페이스
- `adapters/{vllm,ollama,openai,tgi}_adapter.py`: 백엔드별 구현 (응답 스키마 정규화)
- `adapters/registry.py`: `.env`의 `LLM_BACKEND`에 따라 런타임 선택 → `llm_adapter`
- `model_router.py`: 언어/정책에 따라 **모델명만** 선택 (백엔드 독립)
- LangGraph 노드: `llm_adapter.chat(...)`만 호출 → 백엔드 교체 시 수정 불필요

---

## 포함 파일 목록


---
