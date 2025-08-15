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
pip install -r requirements.txt
python scripts/build_faiss.py --docs docs/*.txt

# 3) 서버 실행
uvicorn server.app:app --reload
```
