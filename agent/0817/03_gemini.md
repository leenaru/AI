네, 요청하신 AI 챗봇의 요구사항 정리 및 시스템 아키텍처 설계를 제안해 드립니다. 온디바이스 우선 처리와 서버 기반의 강력한 기능을 결합한 '하이브리드 AI 에이전트' 모델을 중심으로 설계했습니다.

-----

## 📋 요구사항 명세서

### 1\. 프로젝트 개요

**카메라 프리뷰 위에서 동작하는 '음성-비전 기반 온디바이스 우선 AI 에이전트'** 개발을 목표로 합니다. 사용자는 음성 대화와 카메라 비전을 통해 IoT 기기 설정, 문제 해결, 정보 탐색 등의 업무를 직관적이고 신속하게 처리할 수 있습니다.

### 2\. 핵심 가치 및 원칙

  * **🚀 저지연성 (Low Latency) & 개인정보보호 (Privacy):** 핵심적인 사용자 입력 처리(음성인식, 의도분류)는 디바이스 내에서 우선 수행하여 반응 속도를 극대화하고 민감한 정보의 외부 전송을 최소화합니다.
  * **🌐 오프라인 지원 (Offline-First):** 네트워크 연결이 불안정하거나 끊겼을 때도 기본적인 안내와 데이터 수집이 가능하며, 연결 복구 시 서버와 동기화합니다.
  * **🧩 유연성 및 독립성 (Flexibility & Independence):** 특정 LLM이나 클라우드 벤더에 종속되지 않는 개방형 아키텍처를 지향하여, 향후 기술 스택 변경 및 확장이 용이하도록 설계합니다.
  * **💰 비용 효율성 (Cost-Effectiveness):** 가능한 많은 연산을 온디바이스(무료) 또는 온프레미스(제어 가능) 환경에서 처리하여 외부 API 호출 비용을 최소화합니다.

### 3\. 대표 사용자 시나리오 (User Flow)

1.  **입력 (Input):** 사용자가 앱 내 카메라 프리뷰 화면에서 '말하기 버튼' (Push-to-Talk)을 누르고 음성으로 요청합니다. (예: "이 공유기 설치 좀 도와줘.")
2.  **온디바이스 처리 (On-Device Processing):**
      * **ASR (STT):** 디바이스 내장 음성 인식기가 텍스트로 변환합니다.
      * **1차 의도 분류 (NLU):** 경량 LLM(sLLM) 또는 규칙 기반 엔진이 텍스트를 분석하여 단순 응답, UI 조작, 서버 요청 필요 여부를 판단합니다.
      * **이미지 캡처 및 전처리:** "이거 찍어줘", "상태 확인해줘" 등의 의도가 감지되면, 카메라 프레임을 캡처하여 리사이즈, 압축, 민감정보 익명화(예: QR코드 내 개인정보) 처리를 수행합니다.
3.  **서버 요청 (Server Request):** 복잡한 분석, 정보 검색, RAG(Retrieval-Augmented Generation) 등이 필요하다고 판단되면, 텍스트와 전처리된 이미지를 서버로 전송합니다.
4.  **서버 처리 (Server-Side Orchestration):**
      * **LangGraph 실행:** FastAPI로 전달받은 요청을 LangGraph 오케스트레이터가 받아 처리 계획을 수립합니다.
      * **도구 호출 (Tool Calling):** 계획에 따라 Vision 모델(이미지 분석), RAG(매뉴얼/지식베이스 검색), 외부 API(호환성 체크), DB(사용자/기기 정보 조회) 등 적절한 도구를 순차적 또는 병렬적으로 호출합니다.
      * **응답 생성:** 도구 호출 결과를 종합하여 최종 답변을 생성하고, 필요시 다국어로 번역합니다.
5.  **결과 출력 (Output):**
      * **음성 안내 (TTS):** 서버에서 받은 텍스트 응답을 디바이스의 TTS 엔진이 음성으로 변환하여 출력합니다.
      * **시각적 피드백 (UI Overlay):** 음성 안내와 함께 카메라 프리뷰 위에 체크리스트, 가이드 화살표, 확인 버튼 등 관련 UI를 오버레이하여 사용자의 다음 행동을 명확하게 유도합니다.
6.  **오프라인 처리 (Offline Handling):** 서버 요청 중 네트워크 오류가 발생하면, 요청 데이터를 로컬 큐(Queue)에 저장하고 일정 시간 간격으로 재전송을 시도합니다. 연결이 복구되면 큐에 쌓인 요청을 순차적으로 처리합니다.

-----

## 🏛️ 시스템 아키텍처 설계

### 1\. 클라이언트 아키텍처 (On-Device) 📱

클라이언트는 빠른 반응과 오프라인 기능을 담당하는 핵심 영역입니다.

  * **① 입력 계층 (Input Layer):**
      * **Voice Input:** `Push-to-Talk` UI, Android `SpeechRecognizer` / iOS `Speech` framework를 활용한 음성 입력.
      * **Vision Input:** `CameraX` (Android) / `AVFoundation` (iOS)을 통한 실시간 카메라 프리뷰.
  * **② 처리 계층 (Processing Layer):**
      * **ASR (STT) Engine:** OS 내장 음성 인식기를 사용하여 텍스트 변환.
      * **NLU Engine (Intent Classifier):** 1차 의도 분류를 담당.
          * **Rule-based:** "사진 첨부", "도움말" 등 명확하고 빈번한 명령어는 정규식/키워드 매칭으로 즉시 처리.
          * **On-device sLLM:** `Gemma`, `Phi-3 Mini` 등 경량 모델을 `ML Kit`, `CoreML`을 통해 탑재하여, 규칙 기반으로 처리하기 어려운 문맥적 의도(예: "연결이 자꾸 끊기는데 왜 이러지?")를 분류.
      * **Image Pre-processor:** `OpenCV` 또는 플랫폼 네이티브 라이브러리를 사용해 이미지 리사이즈, 압축, OCR을 통한 텍스트 추출 및 익명화 수행.
      * **Offline Handler & Queue:** `Room` (Android) / `CoreData` (iOS) 같은 로컬 DB를 활용하여 네트워크 단절 시 요청 데이터를 저장하고, `WorkManager` (Android) / `BackgroundTasks` (iOS)를 통해 재전송 로직 관리.
  * **③ 출력 계층 (Output Layer):**
      * **TTS Engine:** OS 내장 TTS 엔진으로 텍스트를 자연스러운 음성으로 출력.
      * **UI Overlay Manager:** 서버 응답에 포함된 UI 메타데이터(예: `{ "type": "checklist", "items": ["전원 연결", "WPS 버튼 누르기"] }`)를 파싱하여 카메라 화면 위에 네이티브 UI 컴포넌트를 렌더링.

### 2\. 서버 아키텍처 (On-Premise / Cloud) ☁️

서버는 복잡한 추론, 데이터베이스 연동, 외부 정보 접근 등 고수준의 작업을 처리합니다.

  * **① API 게이트웨이 (API Gateway):**
      * **`FastAPI`:** Python 기반의 비동기 웹 프레임워크로 클라이언트와의 모든 통신(HTTP/WebSocket)을 담당. 요청 유효성 검사, 인증/인가 처리.
  * **② 오케스트레이션 엔진 (Orchestration Engine):**
      * **`LangGraph`:** 상태(State)를 유지하며 복잡하고 순환적인 에이전트 워크플로우를 구성. 사용자의 다단계 문제 해결(예: 기기 등록 트러블슈팅) 시나리오에 최적화. 그래프의 각 노드(Node)는 특정 도구를 호출하거나 LLM 추론을 수행.
  * **③ 핵심 모델 및 도구 (Models & Tools):**
      * **Core LLM:** `Llama 3`, `Mixtral` 등 자체 호스팅이 가능한 고성능 LLM. LangGraph의 노드 내에서 계획 수립, 최종 응답 생성, 사용자 대화 요약 등을 담당.
      * **Vision Agent:** 멀티모달 모델(`LLaVA`, `Florence-2` 등)을 활용하여 클라이언트에서 받은 이미지를 분석. (예: 기기 모델명 식별, LED 상태 색상 인식, 케이블 연결 포트 확인)
      * **RAG Agent (Retrieval-Augmented Generation):**
          * **Knowledge Base:** 제품 매뉴얼, FAQ, 과거 VOC 데이터, 트러블슈팅 가이드 등을 사전에 임베딩하여 `Vector DB`에 저장.
          * **Retriever:** 사용자 질문과 가장 관련성 높은 문서를 Vector DB에서 검색.
          * **Generator:** 검색된 문서를 컨텍스트로 Core LLM에 전달하여 정확하고 근거 있는 답변 생성.
      * **Database Agent:** SQL 데이터베이스에 저장된 사용자 정보, 등록된 기기 목록 및 상태 등을 조회/수정하는 도구.
      * **External API Agent:** 외부 서비스 API를 호출하는 도구. (예: 신제품 정보 크롤링, 제조사 서버와 통신하여 호환성 정보 확인)
  * **④ 데이터 저장소 (Data Stores):**
      * **Vector DB:** `ChromaDB`, `FAISS`, `Weaviate` 등 RAG를 위한 벡터 데이터베이스.
      * **Relational DB:** `PostgreSQL`, `MySQL` 등 사용자 계정, 기기 정보, 대화 로그 등 정형 데이터 저장.

### 3\. 기술 스택 제안 (Tech Stack)

| 구분              | 기술                                                                                              | 역할                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Client (Android)** | Kotlin, CameraX, SpeechRecognizer, ML Kit, Room, WorkManager                                    | 앱 로직, 카메라/음성 입출력, 온디바이스 ML, 오프라인 처리 |
| **Client (iOS)** | Swift, AVFoundation, Speech, CoreML, CoreData, BackgroundTasks                                  | 앱 로직, 카메라/음성 입출력, 온디바이스 ML, 오프라인 처리 |
| **Server** | Python, FastAPI, LangGraph, Uvicorn/Gunicorn                                                      | API 서버, AI 오케스트레이션, 비동기 처리                   |
| **LLM / Vision** | **On-Device:** Gemma, Phi-3 Mini \<br\> **Server:** Llama 3, Mixtral \<br\> **Vision:** LLaVA             | 의도 분류, 계획 수립, 응답 생성, 이미지 분석             |
| **RAG** | `sentence-transformers` (임베딩), `ChromaDB` / `FAISS` (Vector DB)                                  | 지식베이스 검색 및 증강 생성                               |
| **Database** | PostgreSQL / MySQL                                                                                | 사용자, 기기, 로그 등 정형 데이터 관리                    |

이 아키텍처는 초기에는 핵심 기능(기기 등록 가이드) 중심으로 MVP를 구축하고, 점차 트러블슈팅, VOC 대응 등 고도화된 기능으로 확장해 나가는 단계적 개발에 적합합니다.
