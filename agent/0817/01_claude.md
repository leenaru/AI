# IoT AI 챗봇 시스템 아키텍처 설계

## 1. 요구사항 종합 정리

### 1.1 핵심 목적
- **주요 기능**: IoT 기기 등록, 설정, 문제해결을 위한 음성 대화형 에이전트
- **처리 우선순위**: 온디바이스 → 온프레미스 → 클라우드 (비용 최소화)
- **핵심 가치**: 낮은 지연시간, 개인정보 보호, 오프라인 지원, 벤더 독립성

### 1.2 주요 사용 사례
1. **IoT 기기 등록 지원**: 복잡한 등록 과정 가이드
2. **트러블슈팅**: 능동적 에러 대응 및 사전 요구사항 검증  
3. **사용자 등록 및 관리**
4. **기기 구매 가이드**: 호환성 체크 및 추천
5. **매뉴얼 가이드**: 상황별 사용법 안내
6. **장애 대응**: VOC 처리 및 오프라인 진단

### 1.3 성능 요구사항
- 로컬 반응: < 100ms
- ASR 부분 응답: < 300ms  
- TTS 시작: < 700ms
- 서버 왕복: < 1.5s (p95)

## 2. 전체 시스템 아키텍처
<img width="3866" height="2551" alt="image" src="https://github.com/user-attachments/assets/e88e326b-c8c0-4ae2-b979-c9851803e3f9" />

### 2.1 3계층 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER (모바일 앱)                    │
├─────────────────────────────────────────────────────────────┤
│                   SERVER LAYER (백엔드)                      │
├─────────────────────────────────────────────────────────────┤
│                KNOWLEDGE LAYER (지식 기반)                    │
└─────────────────────────────────────────────────────────────┘
```

## 3. Client Layer (모바일 앱) 아키텍처

### 3.1 온디바이스 컴포넌트

```
┌───────────────────────────────────────────────────────────┐
│                  Mobile App Architecture                   │
├───────────────────────────────────────────────────────────┤
│  UI Layer                                                 │
│  ├─ Camera Preview                                        │
│  ├─ Voice UI (Push-to-Talk)                              │
│  ├─ Action Overlay (Checklist, Buttons)                  │
│  └─ Status Indicators                                     │
├───────────────────────────────────────────────────────────┤
│  Local Processing Layer                                   │
│  ├─ STT Engine (Android: SpeechRecognizer / iOS: Speech) │
│  ├─ TTS Engine (Native Platform TTS)                     │
│  ├─ Small LLM (Gemma 3N for local NLU)                 │
│  ├─ Image Preprocessor (Resize, Compress, Anonymize)     │
│  └─ Local NLU Rules                                      │
├───────────────────────────────────────────────────────────┤
│  Communication Layer                                      │
│  ├─ Network Manager (Queue, Retry, Offline Detection)    │
│  ├─ Encryption Handler                                    │
│  └─ Response Cache                                        │
├───────────────────────────────────────────────────────────┤
│  Device Integration Layer                                 │
│  ├─ Camera Controller                                     │
│  ├─ Sensor Access (Location, Network Status)             │
│  └─ File System Access                                    │
└───────────────────────────────────────────────────────────┘
```

### 3.2 로컬 처리 플로우

```
사용자 음성 입력
       ↓
   STT 처리 (온디바이스)
       ↓
   로컬 NLU 룰/소형 LLM 분류
       ↓
   ┌─────────────┬─────────────┐
   │ 로컬 처리 가능 │ 서버 처리 필요 │
   │             │             │
   ↓             ↓             │
TTS 직접 응답    서버 요청 전송    │
                       ↓        │
                  응답 수신       │
                       ↓        │
                  TTS 출력      │
                              │
오프라인 시 → 큐잉 후 재전송 ←─────┘
```

## 4. Server Layer 아키텍처

### 4.1 전체 서버 구조

```
┌──────────────────────────────────────────────────────────────┐
│                     API Gateway                              │
│              (FastAPI + Authentication)                      │
├──────────────────────────────────────────────────────────────┤
│                 LangGraph Orchestrator                       │
│  ┌────────────┬──────────────┬─────────────┬───────────────┐ │
│  │   Policy   │   Planning   │ Tool Exec   │   Response    │ │
│  │  Manager   │    Agent     │   Engine    │  Generator    │ │
│  └────────────┴──────────────┴─────────────┴───────────────┘ │
├──────────────────────────────────────────────────────────────┤
│                 Model Adapter Layer                          │
│  ┌─────────────┬─────────────┬─────────────┬──────────────┐  │
│  │    vLLM     │   Ollama    │ Remote LLM  │   Fallback   │  │
│  │  (로컬GPU)    │  (CPU추론)   │ (클라우드)    │   Router     │  │
│  └─────────────┴─────────────┴─────────────┴──────────────┘  │
├──────────────────────────────────────────────────────────────┤
│                    Tool Services                             │
│  ┌─────────────┬─────────────┬─────────────┬──────────────┐  │
│  │  Vision AI  │ GraphRAG+   │ Translation │   External   │  │
│  │   Service   │   FAISS     │   Manager   │    APIs      │  │
│  └─────────────┴─────────────┴─────────────┴──────────────┘  │
├──────────────────────────────────────────────────────────────┤
│                   Data & Storage                             │
│  ┌─────────────┬─────────────┬─────────────┬──────────────┐  │
│  │ PostgreSQL  │    Redis    │   FAISS     │   MinIO      │  │
│  │(메타데이터)   │  (캐시)      │ (벡터DB)     │ (파일저장)     │  │
│  └─────────────┴─────────────┴─────────────┴──────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 LangGraph 상태 기계 설계

```python
# LangGraph 워크플로우 상태 정의
class ConversationState(TypedDict):
    user_input: str
    intent: str
    context: dict
    tools_needed: List[str]
    current_step: str
    conversation_history: List[dict]
    device_info: dict
    error_state: Optional[str]

# 주요 노드 정의
nodes = {
    "intent_classifier": classify_user_intent,
    "policy_checker": check_compliance_policies, 
    "planner": create_execution_plan,
    "vision_analyzer": analyze_images,
    "rag_retriever": search_knowledge_base,
    "translator": handle_multilingual,
    "response_generator": generate_final_response,
    "action_executor": execute_device_actions
}
```

### 4.3 Model Adapter Layer

```
┌─────────────────────────────────────────────────────────────┐
│                Model Adapter Layer                          │
├─────────────────────────────────────────────────────────────┤
│  Model Router                                               │
│  ├─ Cost-based routing (저비용 모델 우선)                      │
│  ├─ Quality threshold validation                            │
│  ├─ Fallback chain (local → remote)                        │
│  └─ Load balancing                                          │
├─────────────────────────────────────────────────────────────┤
│  Inference Backends                                         │
│  ├─ vLLM Server (GPU): Llama 3, Mixtral                   │
│  ├─ Ollama (CPU): Gemma 7B, Qwen                          │
│  ├─ Remote APIs: GPT-4, Claude (fallback only)            │
│  └─ Specialized: Whisper (ASR), FastEmbed (embeddings)     │
├─────────────────────────────────────────────────────────────┤
│  Optimization Layer                                         │
│  ├─ Token diet (prompt compression)                        │
│  ├─ Semantic cache                                          │
│  ├─ Batch processing                                        │
│  └─ Response streaming                                      │
└─────────────────────────────────────────────────────────────┘
```

## 5. Knowledge Layer 아키텍처

### 5.1 GraphRAG + Vector DB 구조

```
┌─────────────────────────────────────────────────────────────┐
│                   Knowledge Base Layer                      │
├─────────────────────────────────────────────────────────────┤
│  Document Processing Pipeline                               │
│  ├─ Crawlers (IoT manuals, FAQs, troubleshooting docs)    │
│  ├─ ETL Pipeline (clean, chunk, metadata extraction)       │
│  ├─ Embedding Generator (sentence-transformers)            │
│  └─ Graph Builder (entity/relationship extraction)         │
├─────────────────────────────────────────────────────────────┤
│  Storage & Indexing                                         │
│  ├─ FAISS Vector Index (semantic search)                   │
│  ├─ NetworkX Graph (entity relationships)                  │
│  ├─ PostgreSQL (metadata, versions, source tracking)       │
│  └─ MinIO (original documents, images)                     │
├─────────────────────────────────────────────────────────────┤
│  Retrieval Engine                                          │
│  ├─ Hybrid Search (vector + keyword + graph)              │
│  ├─ Reranking (cross-encoder models)                       │
│  ├─ Context Assembly                                        │
│  └─ Source Attribution                                      │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 지식 베이스 관리

```
지식 소스 관리:
├─ IoT 기기 매뉴얼 (PDF, HTML)
├─ FAQ 데이터베이스
├─ 트러블슈팅 가이드
├─ 호환성 매트릭스
├─ 에러 코드 사전
└─ 업데이트 로그

버전 관리:
├─ Git-based versioning
├─ Incremental updates
├─ A/B testing for knowledge
└─ Quality metrics tracking
```

## 6. 데이터 플로우 및 상호작용

### 6.1 전체 요청 처리 플로우

```
1. 사용자 음성 입력 (모바일)
   ↓
2. 온디바이스 STT + 로컬 NLU
   ↓
3. 의도 분류 (로컬 처리 vs 서버 처리)
   ↓
4. [서버 처리 필요시] 암호화된 요청 전송
   ↓
5. API Gateway → LangGraph Orchestrator
   ↓
6. Policy Check → Planning → Tool Execution
   ↓ 
7. Knowledge Retrieval (GraphRAG + Vector Search)
   ↓
8. Response Generation (다국어 번역 포함)
   ↓
9. 응답 반환 (JSON + 액션 지시사항)
   ↓
10. 클라이언트 TTS + UI 업데이트
    ↓
11. [필요시] 사용자 액션 수행 → 결과 회신
```

### 6.2 오프라인/네트워크 복원 처리

```
네트워크 단절 감지
        ↓
로컬 큐에 요청 저장
        ↓
제한적 로컬 응답 제공
        ↓
백그라운드 연결 체크
        ↓
네트워크 복원 감지
        ↓
큐 내용 순차 전송
        ↓
서버 응답 처리
```

## 7. 비용 최적화 전략

### 7.1 토큰 및 연산 비용 최소화

```
온디바이스 우선 처리:
├─ 간단한 FAQ → 로컬 룰 기반 응답
├─ 반복 질문 → 의미 캐시 활용
├─ 기본 설정 → 템플릿 기반 응답
└─ 인사/안부 → 하드코딩 응답

서버 호출 최적화:
├─ 프롬프트 압축 (핵심 정보만)
├─ Max tokens 제한
├─ Stop sequences 엄격 적용
└─ 배치 처리 (가능한 경우)

모델 라우팅:
├─ Gemma 3N (로컬) → Llama 3 8B → GPT-4 (fallback)
├─ 품질 임계값 기반 자동 선택
└─ 실패시 상향 페일오버
```

### 7.2 인프라 비용 최적화

```
스토리지 계층화:
├─ Hot data: 로컬 SSD (FAISS 인덱스)
├─ Warm data: PostgreSQL (메타데이터)
└─ Cold data: MinIO/S3 (원본 문서)

연산 리소스 관리:
├─ GPU: vLLM 전용 (고성능 추론)
├─ CPU: Ollama (경량 모델)
└─ 오토스케일링 (수요 기반)

배치 처리:
├─ 임베딩 생성: 야간 배치
├─ 인덱스 업데이트: 증분 처리
└─ 로그 분석: 오프피크 시간
```

## 8. 보안 및 프라이버시

### 8.1 데이터 보호 전략

```
데이터 최소화:
├─ PII 자동 마스킹 (로컬)
├─ 이미지 익명화 처리
├─ 로그 개인정보 제거
└─ 필수 정보만 서버 전송

암호화:
├─ 전송: TLS 1.3
├─ 저장: AES-256
├─ 키 관리: HashiCorp Vault
└─ End-to-end 암호화 옵션

접근 제어:
├─ 사용자 인증 (JWT)
├─ API 키 관리
├─ 역할 기반 접근 제어
└─ 감사 로그
```

### 8.2 리텐션 정책

```
데이터 보관 기간:
├─ 대화 로그: 30일 (분석 후 삭제)
├─ 이미지 데이터: 7일 (처리 후 삭제)
├─ 메타데이터: 1년 (익명화)
└─ 에러 로그: 90일

사용자 제어:
├─ 데이터 삭제 요청
├─ 처리 내역 조회
├─ 동의 철회 옵션
└─ 데이터 포팅 지원
```

## 9. 모니터링 및 관측성

### 9.1 메트릭 및 알림

```
성능 메트릭:
├─ 응답 시간 (p50, p95, p99)
├─ 처리량 (requests/sec)
├─ 에러율 (4xx, 5xx)
└─ 리소스 사용률

비즈니스 메트릭:
├─ 대화 완료율
├─ 사용자 만족도
├─ 기기 등록 성공률
└─ 문제 해결률

비용 메트릭:
├─ 토큰 사용량
├─ 모델별 호출 비용
├─ 인프라 비용
└─ 캐시 히트율
```

### 9.2 로깅 및 트레이싱

```
분산 트레이싱:
├─ OpenTelemetry 기반
├─ 요청별 추적 ID
├─ 서비스간 의존성 추적
└─ 병목 지점 식별

로그 관리:
├─ 구조화된 로깅 (JSON)
├─ 중앙집중식 수집
├─ 실시간 분석
└─ 개인정보 자동 마스킹
```

## 10. 배포 및 운영

### 10.1 컨테이너 오케스트레이션

```yaml
# Kubernetes 배포 예시
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langraph-orchestrator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langraph-orchestrator
  template:
    spec:
      containers:
      - name: orchestrator
        image: iot-chatbot/orchestrator:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi" 
            cpu: "2000m"
```

### 10.2 CI/CD 파이프라인

```
개발 워크플로우:
├─ 코드 커밋 → 자동 테스트
├─ 도커 이미지 빌드
├─ 보안 스캔 (SAST/DAST)
├─ 스테이징 배포
├─ 통합 테스트
└─ 프로덕션 배포

모델 배포 파이프라인:
├─ 모델 검증 (정확도, 지연시간)
├─ A/B 테스트 설정
├─ 단계적 배포 (canary)
└─ 모니터링 및 롤백 준비
```

## 11. 라이선스 준수

### 11.1 오픈소스 라이선스 전략

```
허용 라이선스:
├─ MIT: React Native, FastAPI
├─ Apache 2.0: LangChain, Kubernetes
├─ BSD-3-Clause: PostgreSQL, NumPy  
└─ AGPLv3: (필요시만) 일부 ML 모델

금지 라이선스:
├─ BSL (Business Source License)
├─ SSPL (Server Side Public License)
├─ RSAL (Redis Source Available License)
└─ 상용 전용 라이선스
```

이 아키텍처는 요구사항을 충족하면서 확장 가능하고 비용 효율적인 IoT AI 챗봇 시스템을 구현할 수 있도록 설계되었습니다. 온디바이스 처리 우선, 다중 모델 지원, 강력한 오프라인 기능을 통해 안정적이고 효율적인 사용자 경험을 제공할 것입니다.
