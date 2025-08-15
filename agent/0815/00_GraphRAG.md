아래에 **요약 → 전체 절차(로컬/Python · Docker Compose · Helm/K8s)** 순서로 정리했습니다.
(이미 패키지/차트/컴포즈 스켈레톤은 이전 답변에서 드린 파일 그대로 사용합니다. 필요하시면 다운로드 링크를 다시 넣어두었습니다.)

---

## 요약

* **목표**: 우리 프로젝트의 **GraphRAG + FAISS 하이브리드 서비스**를 **로컬**, **Docker Compose**, \*\*Kubernetes(Helm)\*\*에서 **배포·구동**하고, **인덱싱 → 질의 → (근거 부족 시) 티켓 생성 → 승인/반영 → 재인덱싱**까지 운영 흐름을 완성합니다.
* **핵심 포인트**

  1. \*\*임베딩은 1개(단일 기준)\*\*로 유지(FAISS 일관성), **챗 LLM은 vLLM↔Ollama** 선택/전환.
  2. **헬스체크/오토페일오버**를 기본 제공(장애 시 자동 폴백).
  3. 인덱싱은 `graphrag index` + `/admin/reindex`(FAISS 리빌드)로 진행.
* **다운로드(이미 제공한 패키지)**

  * 전체 코드: [ZIP](sandbox:/mnt/data/graphrag-hybrid-full-multiembed.zip) / [TAR.GZ](sandbox:/mnt/data/graphrag-hybrid-full-multiembed.tar.gz)
  * 배포 스켈레톤: [deploy/ (Docker & Helm)](sandbox:/mnt/data/graphrag-hybrid-deploy-skeleton.zip)

---

# A. 로컬(Python)에서 시작하기 — 가장 빠른 확인 경로

### A-0) 사전 준비

```bash
# 1) 패키지 풀기
unzip graphrag-hybrid-full-multiembed.zip
cd graphrag-hybrid-full-multiembed

# 2) 가상환경 + 의존성
python -m venv .venv && source .venv/bin/activate
pip install -r app/requirements.txt

# 3) (선택) GraphRAG CLI 설치
pip install graphrag  # 실패해도 무방, 사내 미러/버전 따라 다를 수 있음
```

### A-1) 환경변수(최소)

```bash
# GraphRAG 프로젝트 루트(문서 넣고 인덱싱할 곳)
export GRAPH_RAG_ROOT=./ragproj
export GRAPH_RAG_BIN=graphrag        # 사내 바이너리면 그 경로/이름 지정

# LLM 라우팅(권장: 페일오버)
export USE_FAILOVER=1
export PRIMARY_LLM=vllm              # vllm|ollama|openai
export SECONDARY_LLM=ollama

# 임베딩(단일 기준)
export EMBED_BACKEND=vllm            # vllm|ollama|openai
export EMBED_MODEL=nomic-ai/nomic-embed-text-v1

# 벡터/리랭크
export USE_FAISS=1
export FAISS_CONFIDENCE_MIN=0.74
```

> **vLLM/Ollama 서버**가 필요합니다.
>
> * vLLM: `python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --port 8001`
> * Ollama: `ollama serve` (미설치 시 공식 가이드대로 설치 후 `ollama pull llama3`)
>   기본 엔드포인트는 코드에 `http://localhost:8001/v1`(vLLM), `http://localhost:11434`(Ollama)로 잡혀 있습니다.
>   주소를 바꾸면 `VLLM_BASE_URL`, `OLLAMA_BASE_URL` 환경변수로 알려주세요.

### A-2) 서버 기동

```bash
uvicorn app.server:app --reload --port 8000
# 또는 페일오버 실행 스크립트
./scripts/run_with_failover.sh
```

### A-3) 문서 투입 & 인덱싱

1. **문서 넣기**: `GRAPH_RAG_ROOT` 아래에 `input/` 디렉터리를 만들고 `.txt/.md/.pdf/.docx/.xlsx` 등 소스 파일을 넣습니다.

   * 예) `./ragproj/input/kb/` 아래에 운영 문서, 트러블슈팅 가이드 등.
2. **GraphRAG 인덱싱**(그래프 생성):

   ```bash
   # 프로젝트가 초기화되어 있지 않다면(조직 표준에 맞춰) 설정 파일/스캐폴딩을 준비 후
   # 아래 명령으로 인덱싱을 수행합니다.
   graphrag index --root "$GRAPH_RAG_ROOT"
   ```

   > 사내에서 사용 중인 GraphRAG CLI(또는 파이프라인 명령)가 다르면 `GRAPH_RAG_BIN`에 맞춰 동일하게 실행됩니다.
3. **FAISS 리빌드**(벡터 인덱스):

   ```bash
   curl -X POST http://localhost:8000/admin/reindex
   ```

   * 이 호출은 GraphRAG 산출물(`output/*.parquet`)을 읽어 **FAISS 인덱스**(`faiss.index`, `faiss_meta.parquet`)를 재생성합니다.

### A-4) 질의 & 에스컬레이션 워크플로

```bash
# 1) 질의 (근거 충분 시 즉시 답변)
curl -sX POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"에러 코드 503 대응 방법?","mode":"auto"}' | jq

# 2) 만약 근거 부족(FAISS 최고 점수 < MIN) 시 => 티켓 자동 생성
# 응답 예시: {"status":"escalated","ticket_id":123,...}

# 3) 담당자 답변 등록 (메타 포함)
curl -sX POST http://localhost:8000/escalations/123/reply \
  -H 'Content-Type: application/json' \
  -d '{
        "answer":"원인: 백엔드 헬스체크 실패. 조치: vLLM 재기동 및 HPA 상한 확장.",
        "author":"ops.lead",
        "topic":"troubleshooting","product":"api-gateway","severity":"high",
        "tags":["503","healthcheck","hpa"],"auth":"internal","trust":0.9
      }' | jq

# 4) 운영팀 승인(=> GraphRAG 인덱싱 + FAISS 리빌드 자동 수행)
curl -sX POST http://localhost:8000/escalations/123/approve -H 'Content-Type: application/json' -d '{"approver":"oncall"}' | jq

# 5) 이제 동일/유사 질의가 오면 방금 승인된 문서가 근거로 검색/출처 표기됩니다.
```

### A-5) 헬스체크/전환

```bash
# 애플리케이션 헬스
curl http://localhost:8000/health
curl http://localhost:8000/llm/health

# LLM 백엔드 전환(환경변수 토글)
./scripts/switch_backend.sh vllm     # 또는 ollama | failover
```

---

# B. Docker Compose로 배포 — 풀스택(앱+vLLM+Ollama) 단일 호스트

### B-0) 준비

```bash
# 배포 스켈레톤만 따로 받으셨다면
unzip graphrag-hybrid-deploy-skeleton.zip
cd deploy/docker

cp .env.example .env
# 필요 시 .env에서 USE_FAILOVER/PRIMARY_LLM/SECONDARY_LLM 등 조정
```

### B-1) 빌드/실행

```bash
docker compose up -d --build
# 최초 1회 Ollama 모델 다운로드
docker exec -it ollama ollama pull llama3
```

* 앱은 `http://localhost:8000`(노출 포트),
  vLLM은 `http://localhost:8001/v1`,
  Ollama는 `http://localhost:11434`로 접근됩니다.

### B-2) 인덱싱 & 재인덱싱

* **문서 투입**: `ragproj` 볼륨이 `/data/ragproj`에 마운트되어 있습니다.
  호스트에서 `docker volume inspect`로 실제 경로를 확인하거나, 컨테이너 내부로 복사하세요.

```bash
# 앱 컨테이너 안에서 GraphRAG 인덱싱
docker exec -it graphrag-app bash -lc 'graphrag index --root /data/ragproj'

# FAISS 리빌드
curl -X POST http://localhost:8000/admin/reindex
```

### B-3) 질의/승인/반영은 로컬과 동일

```bash
curl -sX POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{"question":"503 원인?"}'
# ... escalate -> reply -> approve 동일
```

### B-4) 헬스/로그/전환

```bash
docker compose ps
docker logs -f graphrag-app

# 앱 엔드포인트
curl http://localhost:8000/llm/health

# vLLM/Ollama 헬스
curl http://localhost:8001/v1/models
curl http://localhost:11434/api/tags
```

---

# C. Kubernetes(Helm)로 배포 — 실제 운영/확장 전개

### C-0) 앱 이미지 로딩(로컬 클러스터: kind 예시)

```bash
# 루트(프로젝트)에서 앱 이미지 빌드
docker build -f deploy/docker/Dockerfile.app -t graphrag-hybrid-app:local .

# kind로 로컬 클러스터 사용 시
kind load docker-image graphrag-hybrid-app:local
```

### C-1) Helm 설치

```bash
helm install graphrag ./deploy/charts/graphrag-hybrid -n ai --create-namespace
```

* `values.yaml`에서 **백엔드 토글**(vLLM/Ollama), **GPU 사용 여부**, **PVC 크기** 등을 조정하세요.
* 앱의 `VLLM_BASE_URL` / `OLLAMA_BASE_URL`은 템플릿에서 **자동 주입**됩니다.

### C-2) Ollama 모델 1회 pull

```bash
kubectl exec -it deploy/graphrag-hybrid-ollama -n ai -- ollama pull llama3
```

### C-3) 문서 투입 & 인덱싱

* PVC로 마운트된 경로:

  * `ragproj` → `/data/ragproj`
  * `escalations` → `/data/escalations`

```bash
# 앱 파드 내부 진입
kubectl exec -it deploy/graphrag-hybrid -n ai -- bash
# 문서 복사 후 인덱싱
graphrag index --root /data/ragproj
exit

# FAISS 리빌드
kubectl port-forward svc/graphrag-hybrid -n ai 8000:8000
curl -X POST http://localhost:8000/admin/reindex
```

### C-4) 질의/승인/반영

```bash
curl -sX POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"장애 503 조치 가이드?", "mode":"auto"}' | jq

# escalate → reply → approve 루틴은 앞과 동일
```

### C-5) 가시성/운영 팁

* **레디니스/라이브니스**: 차트에 내장되어 있어 롤링/자가복구에 활용됩니다.
* **오토스케일**: HPA/VPA/Cluster Autoscaler는 클러스터 기준으로 추가(필요 시 별도 설치/설정).
* **Ingress**: `values.yaml`에서 `ingress.enabled: true`로 켠 뒤 호스트/TLS 설정.
* **Slack 알림**: `SLACK_WEBHOOK_URL` 또는 `SLACK_BOT_TOKEN`/`SLACK_CHANNEL`을 ConfigMap/Secret로 주입하면 티켓/승인 단계마다 메시지 발송.

---

## 운영 흐름(엔드-투-엔드)

1. 문서 투입(신규 KB/장애 가이드) → `graphrag index`
2. `/admin/reindex`로 FAISS 갱신
3. 사용자 질의 `/query` → **그래프(전역/로컬) + 벡터(유사 텍스트)** 하이브리드 랭킹 → **문장별 출처** 자동 부착
4. 근거 부족 시 **티켓 자동 생성** → 슬랙 알림
5. 담당자 답변 `/escalations/{id}/reply` → 운영 승인 `/approve`
6. 승인 시 **KB 문서화 → GraphRAG 인덱싱 → FAISS 리빌드** 자동 파이프라인
7. 이후 동일/유사 질문은 **갱신된 근거**로 높은 확률로 즉시 답변

---

## FAQ / 트러블슈팅

* **`graphrag` 명령을 못 찾음**

  * `pip install graphrag` 또는 사내 배포된 CLI 경로를 `GRAPH_RAG_BIN`에 설정하세요.
* **FAISS 차원 불일치 오류**

  * 임베딩 모델을 바꾸면 기존 인덱스와 차원이 달라질 수 있습니다. `/admin/reindex`로 **다시** 생성하세요. (가급적 임베딩은 “하나로 고정” 유지)
* **LLM 헬스 실패**

  * vLLM/Ollama 프로세스 상태와 `VLLM_BASE_URL`/`OLLAMA_BASE_URL`을 확인.
  * 일시 장애를 고려해 **Failover 모드**를 기본으로 사용하세요.
* **문장별 출처가 너무 공격적으로 붙음/덜 붙음**

  * `orchestrator/attribution.py`의 `sim_threshold`/`top_k`를 조정해 기준을 올리거나 내릴 수 있습니다.
* **슬랙 미수신**

  * 방화벽/토큰 권한/채널명(프라이빗 채널의 경우 앱 초대 필요) 확인.

---

## 체크리스트(요약)

* [ ] 임베딩 **단일 기준** 설정(변경 시 반드시 재인덱싱)
* [ ] vLLM/Ollama **엔드포인트 정상**(헬스 OK)
* [ ] GraphRAG `index` 성공 후 `/admin/reindex` 수행
* [ ] `/query` 스모크 테스트, 에스컬레이션→승인 플로 테스트
* [ ] Slack/Webhook(선택) 연결 확인
* [ ] 프로덕션은 Helm 기반으로 **PVC·Ingress·오토스케일·모니터링** 구성

---

필요하시면 **예시 문서 세트(샘플 KB)**, **CI 파이프라인(yaml)**, **Grafana 대시보드**까지 맞춰서 드릴게요.
