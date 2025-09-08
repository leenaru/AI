# 요약 (TL;DR)

* \*\*Google ADK (Agent Development Kit)\*\*는 에이전트 개발을 “소프트웨어 개발처럼” 구조화해 주는 오픈소스 프레임워크입니다. Python/Java를 지원하며, 툴(함수/외부 API/MCP/OpenAPI)·세션·메모리·콜백·관측·배포(Cloud Run, Vertex AI Agent Engine)까지 일관된 개발 경험을 제공합니다. ([Google GitHub][1])
* **모델 무관(Model-agnostic)**: Gemini 최적화이지만 LiteLLM을 통해 **Ollama/vLLM** 등 온프레미스·로컬 모델 연동도 공식 가이드로 지원합니다. ([Google GitHub][2])
* **빠른 시작**: `pip install google-adk` → 예제 에이전트 생성 → `adk web`(브라우저 Dev UI) 또는 `adk run`(터미널)으로 실행. ([Google GitHub][3])
* **툴 생태계**: 내장 Google Search, Vertex AI Search, URL/문서 로딩, 장기메모리, 사용자 선택 등 + LangChain/CrewAI 툴 어댑터 + **MCP**/**OpenAPI** 툴셋 제공. ([Google GitHub][4])
* **워크플로/멀티에이전트**: LLM Agent + 순차/루프/병렬 Workflow Agent를 조합하고 ‘에이전트를 툴처럼’ 사용해 팀 구성을 만들 수 있습니다. ([Google GitHub][5])
* **세션·메모리**: 로컬/In-memory로 시작해서, 필요 시 Vertex AI의 **Sessions & Memory Bank**로 확장(Express Mode 무료 체험 포함). ([Google Cloud][6], [Google GitHub][7])
* **양방향 스트리밍**: Gemini Live API와 연동해 음성/비디오 **bidi-streaming**을 지원합니다(Dev UI에서 마이크 바로 사용 가능). ([Google GitHub][3])
* **관측·평가**: Dev UI 이벤트/Trace, Cloud Trace, Opik/Phoenix 등과의 통합, 내장 **Evaluation** 파이프라인 제공. ([Google GitHub][8], [Comet][9])
* **배포**: `adk deploy cloud_run --with_ui`(간편), 또는 **Vertex AI Agent Engine**으로 완전관리형 운영(세션/스케일/운영 기능). ([Google GitHub][10], [Google Cloud][11])
* **노코드 설정(실험적)**: YAML **Agent Config**만으로도 에이전트/서브에이전트/툴 정의 및 실행 가능(`adk create --type=config`). ([Google GitHub][12])

---

# 1) ADK란 무엇인가

\*\*Google Agent Development Kit(ADK)\*\*는 에이전트를 만들고(도구 사용/계획/대화), 관찰하고(이벤트/트레이싱), 평가하고, 배포하는 전 과정을 표준화한 프레임워크입니다. Gemini에 최적화되어 있지만 구조적으로 **모델·배포 무관**하도록 설계되었습니다. Vertex AI의 **Agent Engine**/Agent Builder와 나란히 쓰이도록 제공됩니다. ([Google GitHub][1], [Google Cloud][13])

---

# 2) 핵심 개념과 아키텍처

* **Agent**: 목표를 가진 실행 단위(대화/비대화 모두). **LLM Agent**와 \*\*Workflow Agent(Sequential/Loop/Parallel)\*\*로 구성하며, 상·하위 에이전트로 **멀티에이전트 팀**을 구성할 수 있습니다. 에이전트를 **툴로 래핑**해 상위 에이전트가 호출하는 패턴도 기본 제공. ([Google GitHub][5])
* **Tool**: 함수/메서드/외부 시스템 호출을 통한 기능 확장. 내장 툴 외에 **Third-party(LangChain/CrewAI)**, **MCP**, **OpenAPI**, **Google Cloud 툴** 도입 가능. 툴셋/도구 컨텍스트/상태 관리를 통해 지능적으로 도구 선택·흐름 제어가 가능합니다. ([Google GitHub][14])
* **Session/State/Memory**: 대화 맥락(이벤트 로그), 현재 턴 상태, 장기 메모리(사용자별)로 구분하고, 로컬에서 시작해 **Vertex AI Sessions/Memory Bank**로 확장할 수 있습니다. ([Google GitHub][15], [Google Cloud][6])
* **Runtime & Dev UI**: `adk web`으로 브라우저 기반 Dev UI(채팅·이벤트·트레이스), `adk api_server`로 로컬 API 서버를 제공합니다. ([Google GitHub][3])
* **Callbacks/Events**: Before/After-Agent, Before/After-Model, Before/After-Tool 등 다양한 훅으로 정책/로깅/안전 점검을 삽입합니다. ([Google GitHub][16])

---

# 3) 설치와 환경 설정

### (1) 설치

```bash
python -m venv .venv && source .venv/bin/activate     # macOS/Linux
pip install google-adk
```

프로젝트 생성 후 `adk web`(Dev UI) 또는 `adk run`(터미널) 실행이 가능합니다. (Python 3.9+, Java 17+) ([Google GitHub][3])

### (2) 모델 인증 (Gemini)

* **Google AI Studio** API Key 사용(로컬 간단) 또는
* **Vertex AI**(프로덕션/자동화), \*\*Express Mode(무료 체험)\*\*를 선택할 수 있습니다. `.env`에 `GOOGLE_GENAI_USE_VERTEXAI`와 키/프로젝트 정보를 설정합니다. ([Google GitHub][3])

### (3) 외부/온프레미스 모델 (LiteLLM)

ADK의 **LiteLlm** 래퍼를 통해 **Ollama/vLLM** 등 OpenAI 호환 엔드포인트를 바로 연결할 수 있습니다(툴 호출을 지원하는 모델 권장). ([Google GitHub][2])

---

# 4) 빠른 시작 (Python 예제)

아래는 **도시 시간/날씨** 두 개의 함수 툴을 가진 최소 예제입니다. 생성 후 `adk web`으로 Dev UI에서 바로 대화·이벤트·트레이스를 확인합니다. ([Google GitHub][3])

```python
# multi_tool_agent/agent.py
from datetime import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent

def get_weather(city: str) -> dict:
    """뉴욕만 임의로 응답하는 예시 툴"""
    if city.lower() == "new york":
        return {"status": "success",
                "report": "Sunny, 25°C / 77°F"}
    return {"status": "error",
            "error_message": f"No weather for {city}"}

def get_current_time(city: str) -> dict:
    if city.lower() != "new york":
        return {"status": "error",
                "error_message": f"No timezone info for {city}"}
    now = datetime.now(ZoneInfo("America/New_York"))
    return {"status": "success",
            "report": now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}

root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description="City time & weather",
    instruction="도시 시간/날씨는 반드시 툴로 조회하고 결과만 요약해 답하라.",
    tools=[get_weather, get_current_time],
)
```

실행:

```bash
# 브라우저 Dev UI
adk web multi_tool_agent

# 터미널 실행
adk run multi_tool_agent
```

(Dev UI의 **Events/Trace** 탭에서 툴 호출·지연시간을 시각적으로 점검할 수 있습니다.) ([Google GitHub][3])

---

# 5) 도구(툴) 사용 매뉴얼

## 5.1 Function Tool (커스텀 함수)

* 타입힌트/Docstring을 기반으로 자동 스키마화되어 **함수 인자**를 모델이 추론해 호출합니다.
* `FunctionTool(func=...)`로 감싸 명시적으로 등록할 수도 있습니다. **ToolContext**를 통해 state/메모리/아티팩트 접근 및 에이전트 흐름 제어도 가능합니다. ([Google GitHub][14])

간단 예시:

```python
from google.adk.tools import FunctionTool
from google.adk.agents import Agent

def check_prime(n: int) -> dict:
    """소수 판별 툴"""
    if n < 2: return {"status": "success", "prime": False}
    for k in range(2, int(n**0.5)+1):
        if n % k == 0: return {"status":"success","prime":False}
    return {"status":"success","prime":True}

agent = Agent(
    model="gemini-2.5-flash",
    tools=[FunctionTool(func=check_prime)],
    instruction="소수 여부는 반드시 check_prime 툴을 호출해 확인하라.",
)
```

(상세 개념 및 Toolset/Context/State 활용은 툴 가이드를 참고하세요.) ([Google GitHub][14])

## 5.2 내장(Built-in) 툴

대표 예:

* `google_search` / `enterprise_web_search` / `vertex_ai_search`
* `url_context` / `load_web_page` / `load_artifacts`
* `get_user_choice` / `exit_loop` / `preload_memory` 등
  ADK API 레퍼런스에 전체 목록과 모듈명이 정리되어 있습니다. ([Google GitHub][4])

## 5.3 서드파티 툴

* **LangChainTool / CrewaiTool**: 기존 생태계의 툴을 래핑하여 재사용. 대규모 도구 자산을 손쉽게 가져올 수 있습니다. ([Google GitHub][17])

## 5.4 **MCP**(Model Context Protocol) 툴셋

* MCP 서버에 연결해 파일시스템·DB·내부 서비스 등 안전한 **도구 네트워크**를 노출할 수 있습니다(권한/정책에 유의). ([Google GitHub][18])

## 5.5 **OpenAPI** 툴셋

* OpenAPI 스펙을 읽어 자동으로 함수 호출 인터페이스를 생성합니다(권한/보안 헤더 주입 가능). ([Google GitHub][19])

---

# 6) 멀티에이전트·워크플로

* **Sequential/Loop/Parallel** Workflow Agent로 제어 흐름을 선언하고, **에이전트-as-툴** 패턴으로 라우팅/전문화/위임을 구성합니다.
* 팀 구조(예: Router → Code Tutor / Math Tutor)를 만들고 서브에이전트에 위임하도록 지시문을 설계합니다. ([Google GitHub][20])

노코드 \*\*Agent Config(YAML)\*\*로도 루트/서브에이전트·툴을 정의하고 `adk web`/`adk run`/`adk api_server`로 곧바로 실행할 수 있습니다(실험 기능). ([Google GitHub][12])

---

# 7) 세션·상태·메모리

* **Session**: 대화 스레드(메시지·툴 호출 기록인 이벤트 포함).
* **State**: 해당 세션에서만 유효한 임시 키-값.
* **Memory**: 사용자 장기기억(세션 간 공유). 로컬에서 시작해 **Vertex AI Sessions/Memory Bank**로 확장 가능합니다. ([Google GitHub][15], [Google Cloud][6])

설정 예(Express Mode API 키로 메모리 서비스 연결):

```bash
adk web path/to/agents \
  --memory_service_uri="agentengine://<YOUR_AGENT_ENGINE_ID>"
```

또는 코드에서 `VertexAiMemoryBankService`를 주입합니다. ([Google GitHub][21])

---

# 8) 스트리밍(음성/비디오) & 라이브

* **Gemini Live API**를 사용하는 모델을 지정하면 Dev UI에서 마이크를 켜고 실시간 음성 대화가 가능합니다. WebSocket/SSE 샘플, 스트리밍 도구 가이드가 제공됩니다. ([Google GitHub][3])

---

# 9) 테스트·관측·평가

* **Dev UI**(이벤트/Trace), `adk api_server`로 로컬 API 테스트. ([Google GitHub][3])
* **Cloud Trace**: `adk deploy cloud_run --trace_to_cloud ...` 플래그로 클라우드 추적 활성화. ([Google GitHub][8])
* **서드파티 관측**: **Comet Opik**(오픈소스/클라우드) 및 **Phoenix**(오픈소스)와 공식 통합 가이드 제공. ([Comet][9], [Google GitHub][22])
* **평가(Evaluate)**: 데이터셋 기반 벤치마킹을 위한 내장 평가 루트(방식 2종) 제공. ([Google GitHub][23])

---

# 10) 배포 옵션

## 10.1 Cloud Run(간편)

```bash
# 최소
adk deploy cloud_run --project $GOOGLE_CLOUD_PROJECT --region $GOOGLE_CLOUD_LOCATION ./your_agent

# Dev UI 포함
adk deploy cloud_run --project $PROJECT --region $REGION --with_ui ./your_agent
```

요구 환경변수, FastAPI 내장 API 서버, 인증 토큰(cURL 테스트) 등 가이드가 제공됩니다. ([Google GitHub][10])

## 10.2 Vertex AI Agent Engine(완전관리형)

* 세션/스케일/운영을 Vertex AI가 관리. ADK로 개발한 에이전트를 그대로 올리고, **관리형 세션** API로 사용할 수 있습니다. ([Google GitHub][24], [Google Cloud][25])

---

# 11) 온프레미스/하이브리드(예: Ollama, vLLM) 팁

### (1) Ollama (LiteLLM)

* 툴 호출을 지원하는 모델을 선택합니다(예: 일부 Qwen/Llama 파생).
* LiteLLM 디버그로 실제 요청을 확인할 수 있습니다.

```python
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

agent = LlmAgent(
    model=LiteLlm(model="qwen2.5-coder:latest", api_base="http://localhost:11434/v1"),
    name="local_agent",
    instruction="툴이 있으면 적극 사용하고, 모호하면 확인 질문."
)
```

(모델 선택/툴 지원 확인 및 vLLM(OpenAI 호환) 연결 예시는 모델 가이드를 참고하세요.) ([Google GitHub][2])

### (2) vLLM (OpenAI 호환 엔드포인트)

* `--enable-auto-tool-choice` 등 **툴 호출 파서** 활성화 플래그를 확인하고, ADK에서는 `LiteLlm(api_base=...)`로 연결합니다(Cloud Run 배포 시 ID 토큰 사용 예시 포함). ([Google GitHub][2])

---

# 12) 실전 예시 모음

## 12.1 검색+요약 에이전트 (Built-in Google Search + URL 로딩)

```python
from google.adk.agents import Agent
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.load_web_page import load_web_page

def summarize(url: str) -> dict:
    # 실제 구현에서는 HTML 파싱/정리 후 요약
    return {"status":"success","summary": f"{url}의 핵심 요약(예시)"}

root_agent = Agent(
  model="gemini-2.5-flash",
  name="researcher",
  instruction=(
    "사실 확인이 필요하면 google_search로 검색하고, "
    "출처 URL은 load_web_page로 열람·요약 후 출처를 함께 제시하라."
  ),
  tools=[GoogleSearchTool(), load_web_page, summarize]
)
```

(Google Search/웹 로딩 내장 툴은 공식 레퍼런스 목록에 포함) ([Google GitHub][4])

## 12.2 OpenAPI 연동 (항공편 상태 예시)

OpenAPI 스펙(JSON/YAML)을 읽어 **자동 툴**을 생성해 호출합니다(보안 헤더/쿼리 파라미터 주입 지원). ([Google GitHub][19])

## 12.3 MCP 툴셋(사내 서비스/FS)

사내 MCP 서버에 접속해 허가된 리소스만 에이전트에게 노출합니다(권한/감사 로깅 정책 권장). ([Google GitHub][18])

## 12.4 멀티에이전트 라우터

Router(상위) → CodeTutor/MathTutor(하위)를 구성하고, 상위 지시문에 ‘어떤 질문을 어느 서브에 위임’할지 규칙을 명시합니다. YAML **Agent Config**로도 구현 가능합니다. ([Google GitHub][20])

---

# 13) 운영 체크리스트

1. **지시문 설계**: 툴 사용 조건, 실패 처리 정책(재시도/포기/질문)을 명시. ([Google GitHub][14])
2. **세션/메모리**: 민감정보 저장 범위·보존기간·익명화 정책 수립(Express Mode/Agent Engine 사용 시 리소스 한도 고려). ([Google Cloud][6], [Google GitHub][7])
3. **툴 보안**: OpenAPI/MCP 툴 인증·권한·레이트리밋·감사 로깅. ([Google GitHub][19])
4. **관측**: Dev UI + Cloud Trace/Opik/Phoenix로 추적·평가 대시보드 구축. ([Google GitHub][8], [Comet][9])
5. **배포 전략**: 초기 Cloud Run(간편), 본운영은 Agent Engine(세션/스케일), 또는 온프레(vLLM/Ollama) 하이브리드. ([Google GitHub][10])
6. **스트리밍/멀티모달**: Live API 지원 모델 확인(오디오/비디오/툴 병행). ([Google GitHub][3])

---

# 14) 자주 쓰는 명령어

* 버전 확인: `adk --version`
* Dev UI 실행: `adk web <agent_dir>`
* 터미널 실행: `adk run <agent_dir>`
* 로컬 API 서버: `adk api_server`
* Cloud Run 배포:
  `adk deploy cloud_run --project $PROJECT --region $REGION --with_ui ./agent_dir`
* Agent Engine 배포: 가이드에 따라 리소스 생성 후 SDK로 세션/실행 관리. ([Google GitHub][3])

---

## 참고 문서 (클릭 가능)

* ADK 공식 문서 홈/퀵스타트/러ntime/툴/세션/평가 등: ([Google GitHub][1])
* 모델/인증(오라마/라이트LLM/vLLM): ([Google GitHub][2])
* 멀티에이전트/워크플로/Agent Config: ([Google GitHub][26])
* 스트리밍(Live API): ([Google GitHub][3])
* Cloud Run/Agent Engine 배포: ([Google GitHub][10])
* Sessions & Memory(개요/Express Mode): ([Google Cloud][6], [Google GitHub][7])
* 관측(Cloud Trace/Opik/Phoenix): ([Google GitHub][8], [Comet][9])
* Agent Builder/Agent Engine 개요(제품 맥락): ([Google Cloud][13])

---

원하시면, **귀하의 온프레미스 하이브리드(예: Ollama + LangGraph + ADK) 구조**로 맞춘 \*\*프로젝트 템플릿(폴더 구조, Docker, FastAPI Gateway, OpenAPI/MCP 툴 샘플, Cloud Run/Agent Engine 배포 스크립트)\*\*까지 바로 작성해 드리겠습니다.

[1]: https://google.github.io/adk-docs/?utm_source=chatgpt.com "Agent Development Kit - Google"
[2]: https://google.github.io/adk-docs/agents/models/ "Models & Authentication - Agent Development Kit"
[3]: https://google.github.io/adk-docs/get-started/quickstart/ "Quickstart - Agent Development Kit"
[4]: https://google.github.io/adk-docs/api-reference/python/?utm_source=chatgpt.com "Agent Development Kit documentation - Google"
[5]: https://google.github.io/adk-docs/agents/?utm_source=chatgpt.com "Agents - Agent Development Kit - Google"
[6]: https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/sessions/overview?utm_source=chatgpt.com "Vertex AI Agent Engine Sessions overview"
[7]: https://google.github.io/adk-docs/sessions/express-mode/?utm_source=chatgpt.com "Vertex AI Express Mode - Agent Development Kit - Google"
[8]: https://google.github.io/adk-docs/observability/cloud-trace/?utm_source=chatgpt.com "Cloud Trace - Agent Development Kit - Google"
[9]: https://www.comet.com/docs/opik/tracing/integrations/adk?utm_source=chatgpt.com "Agent Development Kit | Opik Documentation - Comet"
[10]: https://google.github.io/adk-docs/deploy/cloud-run/ "Cloud Run - Agent Development Kit"
[11]: https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview?utm_source=chatgpt.com "Vertex AI Agent Engine overview"
[12]: https://google.github.io/adk-docs/agents/config/ "Agent Config - Agent Development Kit"
[13]: https://cloud.google.com/vertex-ai/generative-ai/docs/agent-builder/overview?utm_source=chatgpt.com "Vertex AI Agent Builder overview"
[14]: https://google.github.io/adk-docs/tools/ "Tools - Agent Development Kit"
[15]: https://google.github.io/adk-docs/sessions/?utm_source=chatgpt.com "Introduction to Conversational Context: Session, State, and ..."
[16]: https://google.github.io/adk-docs/callbacks/?utm_source=chatgpt.com "Callbacks: Observe, Customize, and Control Agent Behavior"
[17]: https://google.github.io/adk-docs/tools/third-party-tools/?utm_source=chatgpt.com "Third party tools - Agent Development Kit - Google"
[18]: https://google.github.io/adk-docs/tools/mcp-tools/ "MCP tools - Agent Development Kit"
[19]: https://google.github.io/adk-docs/tools/openapi-tools/ "OpenAPI tools - Agent Development Kit"
[20]: https://google.github.io/adk-docs/ "Agent Development Kit"
[21]: https://google.github.io/adk-docs/sessions/memory/?utm_source=chatgpt.com "Memory - Agent Development Kit - Google"
[22]: https://google.github.io/adk-docs/observability/phoenix/?utm_source=chatgpt.com "Phoenix - Agent Development Kit - Google"
[23]: https://google.github.io/adk-docs/evaluate/?utm_source=chatgpt.com "Why Evaluate Agents - Agent Development Kit - Google"
[24]: https://google.github.io/adk-docs/deploy/agent-engine/?utm_source=chatgpt.com "Deploy to Vertex AI Agent Engine - Google"
[25]: https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/use/adk?utm_source=chatgpt.com "Use a Agent Development Kit agent | Generative AI on ..."
[26]: https://google.github.io/adk-docs/agents/multi-agents/?utm_source=chatgpt.com "Multi-Agent Systems in ADK - Google"
