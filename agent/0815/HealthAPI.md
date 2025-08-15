아래에 **요약 → 상세 설명 → 실전 예시 코드** 순서로 정리했습니다.

---

## 요약

* **Health 체크 API**는 애플리케이션과 그 의존성(LLM, DB, 인덱스 등)이 **정상 동작 가능한지 신속히 판단**하기 위한 **머신용 진단 엔드포인트**입니다.
* **언제 사용되나?**

  1. **배포/롤링업데이트 시 트래픽 유입 조건(Ready)**,
  2. **비정상 프로세스 감지 후 재시작(Live)**,
  3. **부팅 직후 초기화 완료 판단(Startup)**,
  4. **로드밸런서/게이트웨이의 백엔드 건강검진**,
  5. **오토 페일오버(주→보조 LLM 전환) 판단**,
  6. **모니터링·알림(관측)**.
* **역할**

  * 단순 “프로세스 살아있음?”부터 \*\*의존성 종합 진단(LLM, 그래프 인덱스, FAISS, 슬랙 웹훅 등)\*\*까지 **신뢰 가능한 신호**를 제공해 **자동화된 운영 결정**(트래픽 전환/재시작/스케일링/알림)에 쓰입니다.
* **우리 프로젝트에서**

  * `/health` : 앱 자체 및 필수 리소스(예: 포트 열림, 기본 라우팅)
  * `/llm/health` : **vLLM → 실패 시 Ollama** 순으로 핑(엔드-투-엔드) → **한쪽이라도 OK면 `degraded`로 서비스 지속**
  * (권장) `/readyz`, `/livez`, `/startupz` 를 분리하여 K8s·LB·CI가 **목적별로** 사용

---

# 1) 왜 필요한가? (역할)

1. **트래픽 게이팅(Ready)**

   * 파드/프로세스가 **요청을 정상 처리할 준비**가 됐는지 판단합니다. 예: GraphRAG 인덱스 로딩, FAISS 메모리 매핑이 끝나야 `ready` 로 간주. 준비 전에는 로드밸런서가 **트래픽을 붙이지 않음**.

2. **자가 치유(Live)**

   * 프로세스가 **행응답 불가/데드락** 상태인지 판단합니다. 실패 시 오케스트레이터(예: Kubernetes)가 **자동 재시작**.

3. **부팅 안정화(Startup)**

   * 컨테이너 부팅 직후 **초기화(모델 warmup, 인덱스 로드)** 동안 너무 빨리 Live/Ready 체크를 시작하면 **오탐**이 납니다. Startup 체크는 초기화가 끝날 때까지 **유예 시간**을 보장.

4. **오토 페일오버**

   * `/llm/health`가 **1차(vLLM)** 를 먼저 확인하고 실패하면 **2차(Ollama)** 를 확인해 **서비스 지속**(degraded). LB나 우리 스위치 스크립트가 **백엔드 전환**에 이 신호를 사용.

5. **배포/롤백 자동화**

   * CI/CD에서 새 버전 배포 후 `readyz`가 녹색이면 **롤아웃 계속**, 아니면 **자동 롤백**.

6. **관측/알림과 연계**

   * health 결과를 **메트릭**으로 노출하여 Grafana/Alertmanager로 **장애 감지·알림**(에러율 급증·P95 지연 ↑와 함께 상관 분석).

---

# 2) 어떤 종류가 있나? (패턴)

* **Liveness(/livez)**: “프로세스가 살아 있나?” 최소 체크. 실패 시 **재시작** 트리거.
* **Readiness(/readyz)**: “요청 받을 준비 됐나?” 의존성 준비(인덱스/네트워크/연결) 포함. 실패 시 **트래픽 차단**.
* **Startup(/startupz)**: “초기화 완료됐나?” 완료 전에는 liveness/readiness 체크를 **유예**.
* **Custom Health(/llm/health)**: 우리처럼 **특정 의존성(LLM 백엔드)** 에 대한 **엔드-투-엔드** 검증.

> 한 엔드포인트에 모든 걸 때려 넣는 대신, **목적별 엔드포인트**를 분리하면 **오탐/과탐**을 줄이고, 운영 행동(재시작 vs 트래픽 차단)을 **정확히** 유도할 수 있습니다.

---

# 3) 언제·누가 호출하나?

* **Kubernetes**: `readinessProbe`, `livenessProbe`, `startupProbe`.
* **로드밸런서/Ingress/게이트웨이**: 백엔드 헬스(정상 노드만 라우팅).
* **CI/CD**: 롤링업데이트/블루그린/카나리 중 **배포 게이트**.
* **외부/내부 스케줄러**: `scripts/healthcheck.py` 같은 **주기적 확인**으로 오토-스위치.
* **관측계**: Prometheus **blackbox-exporter**(HTTP 200 여부), 앱 자체 메트릭(`/metrics`)과 연계.

---

# 4) 설계 가이드(우리 프로젝트 기준)

1. **빠르고 가벼워야**

   * 50\~200ms 내 응답 목표. DB 풀쿼리, 대형 임베딩 호출처럼 **무거운 작업 금지**.
   * LLM 핑은 **경량**(모델 목록, 짧은 system ping)으로.

2. **명확한 상태 표현**

   * `status`: `ok | degraded | fail`
   * `components`: { vllm: ok, ollama: fail, faiss: ok, graphrag: ok, slack: optional }
   * `since`(ISO8601), `version`, `commit` 등 **운영에 도움되는 메타** 포함.

3. **Ready 조건을 신중히**

   * 첫 인덱싱 전이면 **ready=false**(트래픽 유입 금지).
   * LLM 1차 불능이더라도 2차가 살아 있으면 **ready=true, status=degraded**로 서비스 지속.

4. **보안/안정성**

   * 외부 공개용 헤더/응답은 **최소 정보**. 세부 진단은 **내부 네트워크**에서만.
   * **레이트 리밋**/캐싱(수십 초)로 남용 방지.

---

# 5) 실전 예시

## 5-1. FastAPI: `/livez`, `/readyz`, `/startupz`, `/llm/health`

```python
# app/health.py (예시)
import os, time
from fastapi import APIRouter
import httpx
from pathlib import Path

router = APIRouter()
START_AT = time.time()

def faiss_ready() -> bool:
    root = os.getenv("GRAPH_RAG_ROOT","./ragproj")
    return Path(f"{root}/faiss.index").exists() and Path(f"{root}/faiss_meta.parquet").exists()

def graphrag_ready() -> bool:
    # 최소한의 파일/폴더 존재 확인(가볍게)
    root = os.getenv("GRAPH_RAG_ROOT","./ragproj")
    return Path(f"{root}/output").exists() or Path(f"{root}/input").exists()

@router.get("/livez")
def livez():
    return {"status":"ok","uptime_sec": round(time.time()-START_AT,2)}

@router.get("/readyz")
def readyz():
    ok_faiss = faiss_ready()
    ok_rag   = graphrag_ready()
    # LLM은 /llm/health에서 종합 판단. 여기선 핵심 의존성만 체크.
    ready = ok_faiss and ok_rag
    return {"status": "ok" if ready else "fail",
            "components": {"faiss": ok_faiss, "graphrag": ok_rag}}

@router.get("/startupz")
def startupz():
    # 가령 초기화 20초 지나야 다른 체크 시작하도록
    return {"status": "ok" if (time.time()-START_AT) > 20 else "fail"}

@router.get("/llm/health")
def llm_health():
    vllm_url = os.getenv("VLLM_BASE_URL","http://localhost:8001/v1")
    oll_url  = os.getenv("OLLAMA_BASE_URL","http://localhost:11434")
    vllm_ok = oll_ok = False

    try:
        with httpx.Client(timeout=1.5) as c:
            c.get(f"{vllm_url}/models").raise_for_status()
            vllm_ok = True
    except Exception:
        vllm_ok = False

    if not vllm_ok:
        try:
            with httpx.Client(timeout=1.5) as c:
                c.get(f"{oll_url}/api/tags").raise_for_status()
                oll_ok = True
        except Exception:
            oll_ok = False

    if vllm_ok:
        return {"status":"ok","backend":"vllm"}
    elif oll_ok:
        return {"status":"degraded","fallback":"ollama"}
    else:
        return {"status":"fail"}
```

> 위 라우터를 `app.server:app`에 include 하면
>
> * **K8s**: `livenessProbe → /livez`, `readinessProbe → /readyz`, `startupProbe → /startupz`
> * **LB/헬스스크립트**: `/llm/health` 로 **오토 페일오버 판단**

## 5-2. Kubernetes 프로브 예시(Helm values 또는 매니페스트)

```yaml
readinessProbe:
  httpGet: { path: /readyz, port: 8000 }
  initialDelaySeconds: 10
  periodSeconds: 10
livenessProbe:
  httpGet: { path: /livez, port: 8000 }
  initialDelaySeconds: 20
  periodSeconds: 20
startupProbe:
  httpGet: { path: /startupz, port: 8000 }
  failureThreshold: 30
  periodSeconds: 2
```

## 5-3. Docker Compose 헬스체크

```yaml
healthcheck:
  test: ["CMD", "curl", "-fsS", "http://localhost:8000/readyz"]
  interval: 15s
  timeout: 5s
  retries: 12
```

## 5-4. Prometheus + Grafana 연계(요약)

* blackbox-exporter로 `/llm/health`의 **200/타임아웃**을 `probe_success`로 수집 → **“LLM Health”** 패널.
* FastAPI에 `/metrics`(prometheus\_fastapi\_instrumentator) 활성화 → **RPS, 지연, 상태코드**.
* Health 실패율이 임계치 초과 시 Alertmanager로 **슬랙 알림**.

---

# 6) 실무 팁 & 안티패턴

* **안티패턴**

  * Health에서 **무거운 DB/외부 API 호출**로 시스템을 더 아프게 함
  * **모든 의존성 실패 = liveness 실패**로 묶어서 **불필요한 재시작 루프** 유발
  * 초기화 중인데 readiness를 너무 빨리 체크 → **오탐**
* **권장**

  * 목적별 엔드포인트 분리(ready/live/startup/custom)
  * **경량 핑 + 캐싱**(예: `/llm/health` 결과 10\~30초 캐시)
  * `ok/degraded/fail`로 **그레이드 표현** → 페일오버/디그레이드 운용에 유리
  * **관측과 결합**(대시보드·알림)해 추세 감시

---

필요하시면 현재 프로젝트 코드에 **`/readyz`/`/livez`/`/startupz`** 라우트를 바로 추가하는 커밋 패치와, Helm 차트에서 프로브를 자동으로 켜는 `values.yaml` 스니펫도 정리해 드릴게요.
