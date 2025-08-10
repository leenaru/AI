알겠습니다. 아래에 **PlantUML 형식**으로 전체 시스템 구조도를 정리해 드립니다. 복사하여 `*.puml` 파일로 저장하신 뒤 렌더링하시면 됩니다.

# 1) 하이레벨 컴포넌트 다이어그램

```plantuml
@startuml HighLevelComponents
title 전체 시스템 하이레벨 아키텍처

skinparam componentStyle rectangle
skinparam rectangle {
  BackgroundColor White
  BorderColor #444444
}
skinparam package {
  BackgroundColor #f8fafc
  BorderColor #94a3b8
}
skinparam ArrowColor #2563eb

' 클라이언트 영역
package "클라이언트 (Android / iOS)" as CLIENTS {
  [모바일 앱\n(Android/iOS)] as Mobile
  [온디바이스 의도/전처리\nGemma 3n (교체 가능)] as OnDeviceNLP
  [센서/스트림\n음성·카메라 프리뷰] as Sensors
  [액션 오케스트레이터\nOPEN_CAMERA / REQUEST_INFO 등] as Orchestrator
}

' 서버/API 영역
package "API 서버 (FastAPI + LangGraph)" as SERVER {
  [API 게이트웨이\n/v1/chat, /v1/events, /v1/tickets, /v1/rag/*] as API
  [정책 오버레이\n국가/정책별 규칙, PII 마스킹] as Policy
  [LangGraph 오케스트레이션\ninstall/usage/troubleshoot/warranty/service] as LangGraph
  [HQ-RAG 서비스\n검색/근거/인입] as RAG
  [티켓 서비스\n에스컬레이션/해결/상태] as Ticket
  [로깅/모니터링] as Observability
}

' 모델 백엔드 추상화
package "Model Adapter Layer" as ADAPTER {
  [model_router\n(언어/정책 → 모델명)] as ModelRouter
  [Adapter Registry\n(LLM_BACKEND 선택)] as AdapterRegistry
  [vLLM Adapter\n(OpenAI 호환)] as VLLM
  [Ollama Adapter] as Ollama
  [OpenAI Adapter] as OpenAI
  [TGI Adapter] as TGI
}

' RAG 백엔드
package "지식 스토어" as KB {
  [Qdrant (옵션)\n임베딩/검색] as Qdrant
  [인메모리 DocStore] as MemStore
  [임베딩 생성기\nSentence-Transformers] as Emb
}

' 운영자 워크플로우
package "운영자/관리" as OPS {
  [운영자 콘솔\n(티켓 처리/해결 입력)] as Operator
  [그래프 GUI\n(React Flow + YAML)] as GraphGUI
}

' 관계
Mobile --> OnDeviceNLP : 1차 의도 분류/오프라인 처리
Sensors -down-> Orchestrator : 프리뷰 이벤트/사용자 허용
Orchestrator --> API : /v1/chat, /v1/events 호출

API --> Policy : 컨텍스트/규칙 적용
API --> LangGraph : 상태/노드 실행
LangGraph --> RAG : 근거 검색/강제 인용
LangGraph --> ModelRouter : 모델명 선택
ModelRouter --> AdapterRegistry : 백엔드 선택
AdapterRegistry --> VLLM
AdapterRegistry --> Ollama
AdapterRegistry --> OpenAI
AdapterRegistry --> TGI
LangGraph --> Ticket : 에스컬레이션 생성/조회/해결

RAG --> Qdrant : (옵션) 검색/인덱스
RAG --> MemStore : (폴백) 검색/인덱스
RAG --> Emb : 임베딩 요청

Operator --> Ticket : 해결 등록(PATCH)
Operator --> RAG : 근거 문서 인입(/v1/rag/ingest)
GraphGUI --> LangGraph : 그래프 YAML 편집/반영

API --> Observability : 구조화 로그/메트릭
@enduml
```

# 2) 트러블슈팅 시나리오 시퀀스 다이어그램

```plantuml
@startuml TroubleshootingSequence
title 트러블슈팅(정확도 우선 + 에스컬레이션) 시퀀스

actor 사용자 as User
participant "모바일 앱\n(Orchestrator)" as App
participant "온디바이스\nGemma 3n" as LocalNLP
participant "API\n(FastAPI)" as API
participant "LangGraph" as Graph
participant "Policy\n(오버레이)" as Policy
participant "RAG 서비스" as RAG
participant "Model Adapter\n(chat)" as Adapter
participant "Qdrant/메모리" as KB
participant "티켓 서비스" as Ticket
actor "운영자" as Operator

== 질의 ==
User -> App : "에어컨 E5 에러"
App -> LocalNLP : 1차 의도 분석(오프라인)
LocalNLP --> App : 결과(트러블슈팅/불확실)
App -> API : POST /v1/chat {text, product?, policy}

== 서버 처리 ==
API -> Policy : 규칙/PII 마스킹/금지토픽 확인
API -> Graph : 상태 주입 후 실행
Graph -> RAG : retrieve(query, tags=troubleshooting, products)
RAG -> KB : 벡터/키워드 검색
KB --> RAG : 관련 문서 리스트
RAG --> Graph : 근거 + 최소 인용 보장

Graph -> Adapter : chat(model, messages=근거요약+포맷)
Adapter --> Graph : 해결 절차(원인/안전/체크리스트/검증)

alt 근거 부족/확신도 낮음
  Graph -> Ticket : CREATE_TICKET
  Ticket --> Graph : ticket_id
  Graph --> API : 답변 + actions=[CREATE_TICKET]
else 확신 충분
  Graph --> API : 답변 + citations
end

API --> App : 텍스트/음성 답변 + 액션

== 운영자 루프 ==
Operator -> Ticket : 해결 내용 PATCH /tickets/{id}/resolve
Ticket -> RAG : add_to_rag (tags=verified)
RAG -> KB : 인덱싱/업데이트
@enduml
```

# 3) 배포/런타임(Deployment) 다이어그램

```plantuml
@startuml Deployment
title 배포/런타임 구성(예: Docker/K8s)

skinparam componentStyle rectangle

node "모바일 단말" as Device {
  component "모바일 앱\n(Android/iOS)" as Mobile
  component "온디바이스 모델\nGemma 3n" as Gemma
}

node "클러스터/서버" as Cluster {
  node "API Pod" {
    component "FastAPI\n(/v1/*)" as FastAPI
    component "LangGraph Runtime" as LangGraph
    component "Policy Overlay" as Policy
    component "Model Adapter Layer" as Adapter
    component "RAG Service" as RAG
    component "Tickets API" as Tickets
    component "Logs/Tracing Agent" as Obs
  }

  node "LLM 백엔드" {
    component "vLLM (OpenAI 호환)" as vLLM
    component "Ollama" as Ollama
    component "OpenAI API" as OpenAI
    component "TGI" as TGI
  }

  node "지식 스토어" {
    database "Qdrant" as Qdrant
    database "In-Memory DocStore" as Mem
  }

  node "운영/도구" {
    component "운영자 콘솔" as Operator
    component "그래프 GUI (React Flow)" as GraphGUI
  }
}

Mobile -[hidden]-> Gemma
Mobile --> FastAPI : HTTPS /v1/chat, /v1/events
Gemma <-down-> Mobile : 의도분류/프리뷰 분석(로컬)

FastAPI --> Policy
FastAPI --> LangGraph
LangGraph --> Adapter
Adapter --> vLLM
Adapter --> Ollama
Adapter --> OpenAI
Adapter --> TGI

LangGraph --> RAG
RAG --> Qdrant
RAG --> Mem

FastAPI --> Tickets
Operator --> Tickets : 해결 입력
Operator --> RAG : /v1/rag/ingest
GraphGUI --> LangGraph : YAML 그래프 반영

FastAPI --> Obs : 로그/메트릭/트레이스
@enduml
```

원하시면 위 세 다이어그램을 **하나의 파일**로 합치거나, 특정 컴포넌트만 상세화한 **서브 다이어그램**(예: Model Adapter Layer 내부 구조 상세, HQ‑RAG 파이프라인 상세)도 추가로 작성해 드리겠습니다.
