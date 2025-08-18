# LangGraph basic example

아래 답변은 **요약 → 상세 설명/코드** 순서로 구성했습니다.
요청하신 대로 **LangGraph + Ollama** 환경에서 **Llama 3**와 **Qwen3-8B**(또는 가용 시리즈) 모델을 번갈아 쓰는 예제를 제공합니다. 모든 코드는 **로컬 온프레미스** 실행을 전제로 합니다.

---

## 요약

* **무엇을**: LangGraph로 대화 상태를 관리하고, Ollama를 통해 **Llama 3**와 **Qwen** 계열(요청: *Qwen3-8B*)을 호출하는 예제 3종

  1. **단일 노드 챗봇** (모델만 바꿔 실행)
  2. **모델 라우팅 그래프** (질문 특성에 따라 Llama3 ↔ Qwen3-8B 선택)
  3. **FastAPI + LangGraph** (간단 REST API; 동기/스트리밍 사용 예시 포함)
* **전제**: `ollama serve`가 동작 중이며, 필요한 모델은 `ollama pull`로 미리 받아둡니다.
* **모델 이름**:

  * Llama 3: 보통 `llama3` 또는 `llama3:instruct`/`llama3:8b-instruct`
  * Qwen3-8B: **Ollama에 정확히 동일한 태그가 없을 수도 있습니다.** 이 경우, **가장 가까운 instruct 계열**(예: `qwen2.5:7b-instruct`, `qwen2.5:14b-instruct`)로 대체해 실행하세요.
  * 모델 태그 가용성은 환경마다 다릅니다. **정확한 태그는 `ollama run`/`ollama pull` 시 안내를 확인**해 주세요.
* **의존성**: `langgraph`, `langchain`, `langchain-community`, `ollama`(파이썬 클라이언트)
* **포인트**: LangGraph의 `StateGraph` + 메시지 누적(`add_messages`) 패턴, 조건부 에지로 간단 라우터 구현, API 서버 예제까지 제공합니다.

---

# 0) 환경 준비

### (1) 설치

```bash
# Python v3.10+ 권장
pip install -U langgraph langchain langchain-community ollama fastapi uvicorn sse-starlette
```

> 일부 환경에서는 `ChatOllama`가 `langchain_ollama` 패키지로 제공되기도 합니다.
> 그 경우: `pip install -U langchain-ollama` 후 `from langchain_ollama import ChatOllama` 형태를 쓰십시오.
> 아래 예제는 **가장 널리 쓰이는** `langchain_community.chat_models` 경로를 기준으로 작성했습니다.

### (2) Ollama 모델 준비

```bash
# Llama 3 (8B instruct 계열 예)
ollama pull llama3:8b-instruct

# Qwen3-8B (환경에 따라 정확 태그 상이 가능)
# 가용하지 않다면 가까운 instruct 계열로 대체하세요 (예: qwen2.5 시리즈)
ollama pull qwen3:8b-instruct         # <- 가능하면 이걸 사용
# 또는
ollama pull qwen2.5:7b-instruct
# 또는
ollama pull qwen2.5:14b-instruct
```

> **모르면 솔직히**: 제 지식만으로는 현재 사용하시는 Ollama 레지스트리에 `qwen3:8b-instruct` 태그가 항상 존재한다고 장담하기 어렵습니다.
> **대안**: 위와 같이 **가장 가까운 Qwen instruct 시리즈**로 대체하면 LangGraph 연동 코드는 동일하게 동작합니다.

---

# 1) 예제 A — “가장 단순한” LangGraph + Ollama 챗봇

* **핵심**: LangGraph의 상태에 `messages`(누적), `model`(선택된 모델 태그)을 두고 단일 노드에서 ChatOllama 호출.
* **모델 바꾸기**: `inputs["model"]`에 `llama3:8b-instruct` 또는 `qwen3:8b-instruct`(가용 시) 등 전달.

```python
# file: example_a_simple_graph.py
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.chat_models import ChatOllama


# 1) 상태 스키마
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    model: str  # ex) "llama3:8b-instruct" or "qwen3:8b-instruct"


# 2) LLM 호출 노드
def call_llm(state: ChatState):
    model_tag = state.get("model") or "llama3:8b-instruct"
    chat = ChatOllama(model=model_tag, temperature=0.2)  # 온디바이스/온프레미스 호출
    ai_msg = chat.invoke(state["messages"])              # LangChain 메시지 배열을 그대로 전달
    return {"messages": [ai_msg]}                        # add_messages 덕분에 누적됨


# 3) 그래프 구성
builder = StateGraph(ChatState)
builder.add_node("llm", call_llm)
builder.set_entry_point("llm")
builder.add_edge("llm", END)
graph = builder.compile()


# 4) 실행 예시
if __name__ == "__main__":
    # Llama3로 묻기
    out1 = graph.invoke({
        "messages": [HumanMessage(content="한 문장으로 자기소개를 해줘.")],
        "model": "llama3:8b-instruct"
    })
    print("[Llama3]", out1["messages"][-1].content)

    # Qwen3(또는 대체 Qwen)로 묻기
    out2 = graph.invoke({
        "messages": [HumanMessage(content="파이썬에서 데코레이터를 간단하게 예로 설명해줘.")],
        "model": "qwen3:8b-instruct"  # 가용하지 않으면 qwen2.5:7b-instruct 등으로 대체
    })
    print("[Qwen]", out2["messages"][-1].content)
```

**실행**

```bash
python example_a_simple_graph.py
```

---

# 2) 예제 B — “질문 특성에 따라 자동 라우팅” (Llama3 ↔ Qwen 선택)

* **핵심**: `router` 노드에서 **간단한 규칙**으로 모델을 선택하고, 조건부 에지(`add_conditional_edges`)로 분기.
* 규칙 예시

  * **한국어/일반 대화** → Llama3
  * **코딩/영어 용어 비중 높음** → Qwen (코딩/리팩토링 등에 강점 가정)

```python
# file: example_b_router_graph.py
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.chat_models import ChatOllama


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    model: str  # 현재 선택 모델(참고용)


def _llm_call_with(model_tag: str):
    def _node(state: ChatState):
        chat = ChatOllama(model=model_tag, temperature=0.2)
        ai_msg = chat.invoke(state["messages"])
        return {"messages": [ai_msg], "model": model_tag}
    return _node


def router(state: ChatState) -> Literal["to_llama", "to_qwen"]:
    """아주 단순한 휴리스틱 라우터 예시."""
    last = state["messages"][-1].content.lower()

    # 코드/영어 키워드가 많으면 Qwen으로
    codeish = any(k in last for k in [
        "python", "java", "typescript", "refactor", "big-o", "complexity",
        "algorithm", "runtime", "api", "class", "function", "decorator", "regex"
    ])
    # 한국어 일반 질의면 Llama3 (임의 규칙)
    koreanish = any(ch in last for ch in "가나다라마바사아자차카타파하")

    if codeish and not koreanish:
        return "to_qwen"
    return "to_llama"


builder = StateGraph(ChatState)
builder.add_node("llama", _llm_call_with("llama3:8b-instruct"))
builder.add_node("qwen",  _llm_call_with("qwen3:8b-instruct"))  # 가용X면 qwen2.5 instruct로 교체
builder.add_node("router", lambda s: s)  # 상태 그대로 전달(분기 전 통과용)

# 시작점 → router
builder.set_entry_point("router")

# router의 반환 라벨에 따라 분기
builder.add_conditional_edges(
    "router",
    router,
    {
        "to_llama": "llama",
        "to_qwen": "qwen",
    }
)

# 각 모델 노드 종료
builder.add_edge("llama", END)
builder.add_edge("qwen", END)

graph = builder.compile()


if __name__ == "__main__":
    # 1) 한국어 일반 질문 → Llama3 쪽 분기 예상
    out_kor = graph.invoke({"messages": [HumanMessage("서울에서 5살 아이와 주말에 할 수 있는 실내활동 추천해줘")]})
    print("[Routed-KOR]", out_kor["model"], "=>", out_kor["messages"][-1].content[:120], "...")

    # 2) 코드스러운 영어 질문 → Qwen 쪽 분기 예상
    out_code = graph.invoke({"messages": [HumanMessage("In Python, show me a decorator that memoizes a function with TTL.")]})
    print("[Routed-ENG]", out_code["model"], "=>", out_code["messages"][-1].content[:120], "...")
```

---

# 3) 예제 C — FastAPI + LangGraph: 간단 REST API

* `/chat`에 프롬프트와 모델 태그를 POST하면 **그래프 실행 결과**를 반환합니다.
* 참고로 LangGraph의 `graph.stream()`을 쓰면 **토큰/스텝 스트리밍**도 가능하지만, 여기서는 **단순/직관 버전**을 먼저 보여드리고, 아래에 **SSE(서버-전송 이벤트) 기반 스트리밍** 예시도 첨부합니다.

```python
# file: example_c_fastapi.py
from typing import Annotated, TypedDict
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.chat_models import ChatOllama


# --- LangGraph 구성 (단일노드) ---
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    model: str

def call_llm(state: ChatState):
    model_tag = state.get("model") or "llama3:8b-instruct"
    chat = ChatOllama(model=model_tag, temperature=0.2)
    ai_msg = chat.invoke(state["messages"])
    return {"messages": [ai_msg]}

builder = StateGraph(ChatState)
builder.add_node("llm", call_llm)
builder.set_entry_point("llm")
builder.add_edge("llm", END)
graph = builder.compile()


# --- FastAPI 앱 ---
app = FastAPI(title="LangGraph + Ollama Demo")

class ChatRequest(BaseModel):
    prompt: str
    model: str | None = None  # ex) "llama3:8b-instruct" | "qwen3:8b-instruct"

class ChatResponse(BaseModel):
    model: str
    reply: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    inputs = {
        "messages": [HumanMessage(content=req.prompt)],
    }
    if req.model:
        inputs["model"] = req.model
    result = graph.invoke(inputs)
    return ChatResponse(
        model=req.model or "llama3:8b-instruct",
        reply=result["messages"][-1].content
    )
```

**실행**

```bash
uvicorn example_c_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

**호출 예**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"요약과 불릿포인트 예제 보여줘", "model":"llama3:8b-instruct"}'
```

### (선택) SSE 기반 토큰 스트리밍

* `graph.stream(inputs)` 또는 LangChain 모델의 `stream()`을 활용해 토큰 단위로 전송하려면 SSE 엔드포인트를 추가하세요.

```python
# file: example_c_streaming.py (핵심 부분만)
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage
# graph는 위와 동일하게 준비되어 있다고 가정

app = FastAPI(title="LangGraph + Ollama Streaming")

@app.get("/stream")
async def stream(prompt: str, model: str = "llama3:8b-instruct"):
    inputs = {"messages": [HumanMessage(content=prompt)], "model": model}

    def event_generator():
        # LangGraph의 스텝/변경 스트림을 그대로 내보내는 예시
        for event in graph.stream(inputs, stream_mode="values"):
            # event는 {"messages":[...]} 꼴; 마지막 메시지 content만 흘려보냄
            try:
                content = event["messages"][-1].content
                yield {"event": "token", "data": content}
            except Exception:
                continue
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(event_generator())
```

프론트엔드(브라우저)에서는:

```javascript
const es = new EventSource("/stream?prompt=Hello&model=llama3:8b-instruct");
es.onmessage = (e) => console.log("chunk:", e.data);
es.addEventListener("done", () => es.close());
```

---

# 4) (옵션) LangGraph 없이 “직접 Ollama 파이썬 클라이언트” 호출

* 일부 환경에서 LangChain 의존성을 줄이고 싶을 때 유용합니다.
* LangGraph로 상태를 관리하되 **노드 내부에서 `ollama.chat()` 직접 호출**도 가능합니다.

```python
# file: example_d_direct_ollama_node.py
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import ollama  # pip install ollama

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    model: str

def call_ollama_direct(state: ChatState):
    model_tag = state.get("model") or "llama3:8b-instruct"
    # LangChain 메시지 → Ollama 포맷 변환
    conv = []
    for m in state["messages"]:
        role = "user"
        if m.type == "ai":
            role = "assistant"
        elif m.type == "system":
            role = "system"
        conv.append({"role": role, "content": m.content})

    res = ollama.chat(model=model_tag, messages=conv, options={"temperature": 0.2})
    ai_text = res.get("message", {}).get("content", "")
    return {"messages": [AIMessage(content=ai_text)]}

builder = StateGraph(ChatState)
builder.add_node("llm", call_ollama_direct)
builder.set_entry_point("llm")
builder.add_edge("llm", END)
graph = builder.compile()

if __name__ == "__main__":
    out = graph.invoke({
        "messages": [HumanMessage(content="이 노드는 LangChain 없이 ollama로 직접 호출합니다. 설명해줘.")],
        "model": "llama3:8b-instruct"
    })
    print(out["messages"][-1].content)
```

---

# 5) 실행/운영 체크리스트

* **Ollama 데몬**: `ollama serve` 가 동작 중인지 확인하세요.
* **모델 태그**: `ollama list`로 로컬에 있는 태그를 점검하고, 본문 코드의 모델명과 일치시키세요.
* **메모리/VRAM**: 8B 계열은 CPU도 가능하지만, GPU/NPU가 있으면 더 빠릅니다.
* **LangGraph 버전**: `add_messages` 임포트 경로가 버전에 따라 `langgraph.graph.message` 또는 `langgraph.types`일 수 있습니다. 오류 시 경로를 바꿔보세요.
* **Qwen3-8B 가용성**: 환경에 없으면 `qwen2.5:7b-instruct` 또는 `qwen2.5:14b-instruct`로 교체 후 테스트하세요.

---

# 6) 자주 묻는 질문(FAQ)

**Q1. Qwen3-8B가 Ollama에서 안 보입니다.**
A. 레지스트리/버전의 차이일 수 있습니다. 같은 Qwen instruct 계열 태그로 대체(예: `qwen2.5:*-instruct`)해서 구조와 코드를 먼저 검증하세요. 이후 환경에서 Qwen3-8B가 제공되면 **모델 태그만** 바꿔 그대로 재사용 가능합니다.

**Q2. 라우팅을 더 똑똑하게 하고 싶습니다.**
A. 간단 규칙 대신 **경량 분류기**(예: 소형 LLM 프롬프트, 키워드+스코어, 토큰 길이, 언어 감지) or **LangGraph tool + 스코어링**으로 확장 가능합니다. 라우팅 결과를 `state["model"]`에 기록해 추적하면 관측/로깅도 용이합니다.

**Q3. 대화 기록을 외부 저장소에 남길 수 있나요?**
A. LangGraph의 체크포인터(예: Redis/PostgreSQL)로 메시지를 영속화할 수 있습니다. 필요하시면 해당 예제도 이어서 제공해 드리겠습니다.

---

필요하시면 **체크포인터 연동**, **툴 호출(예: 검색/DB 조회)**, **멀티턴 상태머신(재질문/확인 단계)**, \*\*모델 별 옵션 튜닝(예: `temperature`, `num_ctx`)\*\*까지 확장한 버전으로도 정리해 드리겠습니다.

---
---
---

# 멀티턴

아래 답변은 **요약 → 상세 코드/설명** 순서로 구성했습니다.
요청하신 대로 **LangGraph를 더 활용**하여 **서너 번의 대화가 자연스럽게 이어지는(슬롯 채우기 + 루프 + 체크포인터)** 예제를 제공합니다. Ollama 기반으로 **Llama 3** 또는 **Qwen(예: Qwen3-8B 가용 시, 없으면 Qwen2.5 instruct 계열)** 중 아무거나 붙여서 동작합니다.

---

## 요약

* **무엇**: LangGraph 상태 머신으로 **여행 추천 멀티턴(3\~4턴)** 대화 구현

  * **부족한 정보(도시/여행일수/아이 동반/예산)** 를 감지 → **부족하면 질문 → 답을 반영 → 다시 점검**을 반복 → **충분해지면 최종 플랜 생성**
  * **조건부 에지(conditional edges)**, **루프(ask → 사용자응답 → 슬롯반영 → 재평가)**, **체크포인터(MemorySaver)** 로 **진짜 멀티턴**을 구현
* **핵심 포인트**

  * 상태 `slots/pending/messages` 를 쓰는 **슬롯-필링 패턴**
  * **엔트리 포인트를 항상 `ingest`** 로 잡아, 매 턴에 **사용자 최신 입력을 흡수 후 라우팅**
  * 모델은 **`state["model"]`** 로 주입: `llama3:8b-instruct` 또는 `qwen3:8b-instruct`(가용 시) / 대체 `qwen2.5:*-instruct`
* **동작 예(4턴)**

  1. 사용자: “7월에 가족이랑 3일 호주 여행 추천” → **도시 물어봄**
  2. 사용자: “시드니요” → **아이 동반 여부 질문**
  3. 사용자: “5살 아이 있어요” → **예산 질문**
  4. 사용자: “중간 정도요” → **최종 일정 생성/답변**

---

# 상세: 멀티턴 LangGraph + Ollama 예제

> 사전 준비
>
> ```bash
> pip install -U langgraph langchain langchain-community ollama
> # (선택) qwen3-8B 태그가 환경마다 다를 수 있습니다. 가용 태그로 대체하세요.
> ollama pull llama3:8b-instruct
> ollama pull qwen3:8b-instruct  # 없으면 아래 중 하나
> # ollama pull qwen2.5:7b-instruct
> # ollama pull qwen2.5:14b-instruct
> ```
>
> `ollama serve`가 실행 중이어야 합니다.

## 파일: `multiturn_slot_filling.py`

```python
# -*- coding: utf-8 -*-
"""
LangGraph + Ollama 멀티턴(슬롯-필링) 예제
- 부족 슬롯을 물어보고, 사용자 답을 반영하며 루프
- 충분해지면 LLM으로 최종 플랜 생성
"""

from __future__ import annotations
import re
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama


# -----------------------------
# 1) 상태 정의
# -----------------------------
class Slots(TypedDict, total=False):
    city: str            # 여행 도시명 (예: 시드니)
    days: int            # 여행 일수 (예: 3)
    with_kid: bool       # 아이 동반 여부
    budget: Literal["low", "mid", "high"]  # 예산 대역

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    slots: Slots
    pending: str | None            # 직전에 물어본 슬롯명 (사용자 최신 답변을 매칭)
    model: str                     # "llama3:8b-instruct" | "qwen3:8b-instruct" | ...
    done: bool                     # 최종 답변 완료 여부


# -----------------------------
# 2) 유틸: 간단 파서(한국어 휴리스틱)
# -----------------------------
def parse_days(text: str) -> int | None:
    # "3일", "3박4일" 등에서 일수 유추 (대략)
    # 우선순위: "(\d+)일"을 찾되 "박" 패턴도 감안
    m = re.search(r"(\d+)\s*일", text)
    if m:
        return int(m.group(1))
    # "3박4일" → 뒤의 4일을 잡을 수 있으니 위 정규식으로 충분
    return None

def parse_with_kid(text: str) -> bool | None:
    # 아이/아기/자녀 등 포함 시 True로 가정, 부정어 포함 시 False 시도
    t = text.lower()
    kor = text
    pos_kw = ["아이", "아기", "자녀", "유아", "초등", "5살", "6살", "7살"]
    neg_kw = ["아이 없음", "아이없", "혼자", "어른만", "성인만"]
    if any(k in kor for k in pos_kw):
        return True
    if any(k in kor for k in neg_kw):
        return False
    # 단순 예/아니오 처리
    if "예" in kor or "네" in kor or "있" in kor:
        return True
    if "아니" in kor or "없" in kor:
        return False
    return None

def parse_budget(text: str) -> Literal["low", "mid", "high"] | None:
    kor = text
    if any(k in kor for k in ["저렴", "가성비", "저가", "절약", "값싼"]):
        return "low"
    if any(k in kor for k in ["중간", "보통", "무난", "중간정도"]):
        return "mid"
    if any(k in kor for k in ["고급", "프리미엄", "럭셔리", "호화"]):
        return "high"
    return None


# -----------------------------
# 3) 노드: 사용자 최신 입력 흡수(슬롯 반영)
# -----------------------------
def ingest(state: ChatState) -> dict:
    """
    매 턴의 엔트리:
    - pending 슬롯이 있으면 사용자의 가장 최신 메시지로 채우기 시도
    - 채우면 pending=None으로 해제
    """
    pending = state.get("pending")
    if not pending:
        return {}  # 이번 턴에 반영할 건 없음

    # 가장 최근 사용자 메시지(마지막 HumanMessage) 추출
    last_user = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user = m
            break
    if not last_user:
        return {}

    text = last_user.content.strip()
    slots = dict(state.get("slots") or {})

    if pending == "city":
        # 단순히 문자열 전체를 도시로 수용(실서비스라면 사전/지오코딩)
        slots["city"] = text
    elif pending == "days":
        d = parse_days(text)
        if d:
            slots["days"] = d
    elif pending == "with_kid":
        w = parse_with_kid(text)
        if w is not None:
            slots["with_kid"] = w
    elif pending == "budget":
        b = parse_budget(text)
        if b:
            slots["budget"] = b

    # 반영이 되었든 아니든, 이번 턴엔 질문 대기 해제
    return {"slots": slots, "pending": None}


# -----------------------------
# 4) 라우터: 부족한 슬롯을 찾아 다음 노드 결정
# -----------------------------
def route_next(state: ChatState) -> Literal["ask_city", "ask_days", "ask_kid", "ask_budget", "plan"]:
    s = state.get("slots") or {}
    # 필요한 슬롯: city → days → with_kid → budget 순서로 채우기
    if not s.get("city"):
        return "ask_city"
    if not s.get("days"):
        return "ask_days"
    if "with_kid" not in s:
        return "ask_kid"
    if not s.get("budget"):
        return "ask_budget"
    return "plan"


# -----------------------------
# 5) 질문 노드들(부족 슬롯을 묻기)
# -----------------------------
def ask_city(state: ChatState):
    msg = AIMessage(content="어느 도시로 여행을 계획 중이신가요? 예: 시드니")
    return {"messages": [msg], "pending": "city"}

def ask_days(state: ChatState):
    msg = AIMessage(content="여행 일정은 며칠로 생각하시나요? 예: 3일")
    return {"messages": [msg], "pending": "days"}

def ask_kid(state: ChatState):
    msg = AIMessage(content="아이와 함께 동행하시나요? (예/아니오) 또는 아이 나이를 알려주세요. 예: 5살 아이 있어요")
    return {"messages": [msg], "pending": "with_kid"}

def ask_budget(state: ChatState):
    msg = AIMessage(content="예산 수준을 알려주세요. (저렴/중간/고급) 중에서 선택해 주세요.")
    return {"messages": [msg], "pending": "budget"}


# -----------------------------
# 6) 최종 플랜 생성 노드(LLM 호출)
# -----------------------------
def plan(state: ChatState):
    s = state["slots"]
    model_tag = state.get("model") or "llama3:8b-instruct"

    sys = SystemMessage(content=(
        "당신은 한국어 여행 플래너입니다. 답변은 공손하고 간결한 요약 → 상세 일정 순서로 제시하세요. "
        "아이 동반 시 유아/아동 친화 활동과 휴식 템포를 포함하세요. "
        "예산(low/mid/high)에 맞춰 식당/교통/티켓 팁을 포함하세요."
    ))
    user = HumanMessage(content=f"""
[요청 파라미터]
- 도시: {s.get('city')}
- 일수: {s.get('days')}일
- 아이 동반: {s.get('with_kid')}
- 예산: {s.get('budget')}

[요청]
- 하루 단위(아침/점심/오후/저녁)로 3~5개 핵심 활동
- 이동 동선 최소화
- 맛집 1~2곳/일, 예약 팁
- 마지막에 '빠른 체크리스트'를 불릿으로 요약
""".strip())

    chat = ChatOllama(model=model_tag, temperature=0.3)
    ai = chat.invoke([sys] + state["messages"] + [user])

    return {"messages": [ai], "done": True}


# -----------------------------
# 7) 그래프 구성 & 체크포인터
# -----------------------------
def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("ingest", ingest)
    builder.add_node("ask_city", ask_city)
    builder.add_node("ask_days", ask_days)
    builder.add_node("ask_kid", ask_kid)
    builder.add_node("ask_budget", ask_budget)
    builder.add_node("plan", plan)

    # 시작 → ingest
    builder.set_entry_point("ingest")
    # ingest 후 상황에 따라 분기
    builder.add_conditional_edges(
        "ingest",
        route_next,
        {
            "ask_city": "ask_city",
            "ask_days": "ask_days",
            "ask_kid": "ask_kid",
            "ask_budget": "ask_budget",
            "plan": "plan",
        },
    )
    # 각 질문 노드는 사용자 응답을 기다리므로 바로 종료
    builder.add_edge("ask_city", END)
    builder.add_edge("ask_days", END)
    builder.add_edge("ask_kid", END)
    builder.add_edge("ask_budget", END)
    # plan은 최종 출력 후 종료
    builder.add_edge("plan", END)

    # 멀티턴을 위한 체크포인터(메모리 저장)
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# -----------------------------
# 8) 데모: 4턴 대화 시뮬레이션
# -----------------------------
if __name__ == "__main__":
    graph = build_graph()
    thread_id = "trip-demo"  # 동일 thread_id로 턴을 이어감
    cfg = {"configurable": {"thread_id": thread_id}}

    def talk(user_text: str, model_tag: str = "llama3:8b-instruct"):
        """사용자 메시지를 그래프에 전달하고 마지막 AI 응답을 출력"""
        res = graph.invoke(
            {
                "messages": [HumanMessage(content=user_text)],
                "model": model_tag,
                "slots": {},
                "pending": None,
                "done": False,
            },
            config=cfg
        )
        # 마지막 AI 메시지 출력
        ai_msg = None
        for m in reversed(res["messages"]):
            if isinstance(m, AIMessage):
                ai_msg = m
                break
        print("\n[Assistant]\n", ai_msg.content if ai_msg else "(AI 응답 없음)")
        return res

    print("=== 턴 1 ===")
    talk("7월에 가족이랑 3일 일정으로 호주 여행 가고 싶어요. 추천해줘.", model_tag="llama3:8b-instruct")

    print("\n=== 턴 2 ===")
    talk("시드니요")

    print("\n=== 턴 3 ===")
    talk("5살 아이 있어요")

    print("\n=== 턴 4 ===")
    talk("중간 정도요")  # 여기서 최종 일정이 생성됩니다.

    # (선택) Qwen으로도 동일 흐름을 새 스레드에서 테스트
    print("\n=== Qwen 스레드(선택) ===")
    cfg_qwen = {"configurable": {"thread_id": "trip-demo-qwen"}}
    res = graph.invoke(
        {
            "messages": [HumanMessage(content="가족과 2일간 도쿄 여행, 예산은 저렴하게. 추천해줘.")],
            "model": "qwen3:8b-instruct",  # 가용X면 qwen2.5 instruct로 교체
            "slots": {},
            "pending": None,
            "done": False,
        },
        config=cfg_qwen
    )
    ai_msg = None
    for m in reversed(res["messages"]):
        if isinstance(m, AIMessage):
            ai_msg = m
            break
    print("\n[Assistant/Qwen]\n", ai_msg.content if ai_msg else "(AI 응답 없음)")
```

---

## 코드 설명 포인트

1. **엔트리 포인트는 항상 `ingest`**

   * 매 턴마다 최신 사용자 입력을 **슬롯에 반영**한 후, **부족 슬롯을 라우팅**합니다.
   * 질문 노드(`ask_*`)는 **assistant 질문만 남기고 종료(END)** → 다음 턴에 사용자가 답하면 `ingest`가 그 답을 흡수합니다.

2. **조건부 에지(라우팅)**

   * `route_next()` 가 `ask_city / ask_days / ask_kid / ask_budget / plan` 중 하나의 **라벨을 반환**하면, `add_conditional_edges` 가 대응 노드로 분기합니다.

3. **체크포인터(MemorySaver)**

   * `compile(checkpointer=MemorySaver())` + `config={"configurable":{"thread_id": ...}}` 로 **대화 맥락 유지**.
   * 실제 서비스에서는 Redis/PostgreSQL 체크포인터로 교체하면 **프로세스 재기동에도 지속**됩니다.

4. **모델 교체 용이**

   * `state["model"]` 로 태그를 주입하므로 **`llama3 ↔ qwen` 교체가 1줄**로 가능합니다.
   * `qwen3:8b-instruct` 태그가 없다면 **`qwen2.5:7b-instruct`** 등 가용 instruct 태그로 바꿔주세요.

5. **간단 파서(휴리스틱)**

   * 예제를 간결히 하려고 정규식/키워드 기반 파서를 넣었습니다.
   * 실제로는 **소형 LLM을 “파서 전용” 노드로** 두거나, **pydantic 모델 + JSON 형식 출력**을 요구하도록 프롬프트를 설계하면 더 견고해집니다.

---

## 확장 아이디어

* **Checkpointer 교체**: `MemorySaver` → Redis/Postgres로 교체해 다중 워커/서버 환경 지원
* **서브그래프**: 도시가 해외면 비자/환전 안내 서브그래프, 국내면 교통카드/숙소 체크 서브그래프
* **스마트 라우터**: 언어 감지/키워드 스코어로 `llama3` ↔ `qwen` 자동 선택 (이전 답변의 router 예제 재활용)
* **스트리밍**: `graph.stream(..., stream_mode="values")` + SSE로 토큰 스트림 전송
* **툴 호출**: 항공/호텔/날씨 API 를 별도 노드로 분리하고 LLM 노드 전/후에 배치

---

원하시면 위 예제를 **FastAPI 엔드포인트**로 감싸 멀티턴을 **HTTP 세션(thread\_id) 기준**으로 이어주는 서버 버전, 또는 **체크포인터를 Redis/PostgreSQL**로 바꾼 실전 템플릿도 바로 드리겠습니다.

---
---
---

# LLM 을 이용한 분기

아래 답변은 **요약 → 전체 코드 → 동작 설명/확장 포인트** 순서로 구성했습니다.
요청하신 대로 **분기(라우팅)를 사전 규칙/문자열이 아니라 LLM의 판단 결과**로 수행하는 LangGraph 예제를 제공합니다. (Ollama 기반 `llama3:8b-instruct` 또는 `qwen3:8b-instruct`/대체 Qwen instruct)

---

# 요약

* **핵심 아이디어**:
  분기를 담당하는 노드(`decide_next`)에서 **LLM에게 현재 슬롯/대화 맥락을 보여주고 “다음에 무엇을 할지”를 JSON으로 답하게** 합니다.
  가능한 액션은 고정된 집합(`ask_city`, `ask_days`, `ask_kid`, `ask_budget`, `plan`)이며, LLM은 이 중 **하나만** 선택합니다.
* **구조**:
  `ingest(슬롯반영) → decide_next(LLM 라우팅) → [ask_* | plan]`
  `ask_*` 노드는 질문만 남기고 `END`. 다음 턴에 유저 답을 `ingest`가 흡수.
* **신뢰성 장치**:

  1. **JSON 강제 프롬프트** + 출력 파싱(파싱 실패 시 안전 기본값),
  2. **허용 액션 화이트리스트** 체크,
  3. **유효하지 않으면 보수적 기본값(가장 먼저 비어있는 슬롯)**.
* **모델 선택**: 상태의 `model`로 주입 → `llama3:8b-instruct` 또는 `qwen3:8b-instruct`(환경에 없으면 `qwen2.5:7b/14b-instruct` 대체).

---

# 전체 코드: `multiturn_llm_branching.py`

```python
# -*- coding: utf-8 -*-
"""
LangGraph + Ollama
LLM 의 판단(JSON) 결과를 사용하여 다음 분기(액션)를 결정하는 멀티턴 예제

필요 패키지:
  pip install -U langgraph langchain langchain-community ollama

사전 준비:
  ollama pull llama3:8b-instruct
  ollama pull qwen3:8b-instruct  # 없으면 qwen2.5:7b-instruct 등 instruct 계열로 대체
  ollama serve  # 백그라운드 실행
"""

from __future__ import annotations
import json
import re
from typing import Annotated, TypedDict, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

# -----------------------------
# 0) 상태 정의
# -----------------------------
AllowedAction = Literal["ask_city", "ask_days", "ask_kid", "ask_budget", "plan"]

class Slots(TypedDict, total=False):
    city: str
    days: int
    with_kid: bool
    budget: Literal["low", "mid", "high"]

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    slots: Slots
    pending: str | None              # 직전에 물어본 슬롯명
    model: str                       # llama3:8b-instruct, qwen3:8b-instruct, ...
    next_action: AllowedAction | None
    reason: str | None               # LLM 라우터의 사유(로깅용)
    done: bool

# -----------------------------
# 1) 간단 파서(한국어 휴리스틱)
# -----------------------------
def parse_days(text: str) -> int | None:
    m = re.search(r"(\d+)\s*일", text)
    if m:
        return int(m.group(1))
    return None

def parse_with_kid(text: str) -> bool | None:
    kor = text
    pos_kw = ["아이", "아기", "자녀", "유아", "초등", "5살", "6살", "7살"]
    neg_kw = ["아이 없음", "아이없", "혼자", "어른만", "성인만"]
    if any(k in kor for k in pos_kw): return True
    if any(k in kor for k in neg_kw): return False
    if any(k in kor for k in ["예", "네", "있"]): return True
    if any(k in kor for k in ["아니", "없"]): return False
    return None

def parse_budget(text: str):
    kor = text
    if any(k in kor for k in ["저렴", "가성비", "저가", "절약", "값싼"]): return "low"
    if any(k in kor for k in ["중간", "보통", "무난", "중간정도"]): return "mid"
    if any(k in kor for k in ["고급", "프리미엄", "럭셔리", "호화"]): return "high"
    return None

# -----------------------------
# 2) ingest: 최신 사용자 입력을 슬롯에 반영
# -----------------------------
def ingest(state: ChatState):
    pending = state.get("pending")
    if not pending:
        return {}
    # 가장 최근 사용자 메시지 찾기
    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user:
        return {}

    text = last_user.content.strip()
    slots = dict(state.get("slots") or {})

    if pending == "city":
        slots["city"] = text
    elif pending == "days":
        d = parse_days(text)
        if d: slots["days"] = d
    elif pending == "with_kid":
        w = parse_with_kid(text)
        if w is not None: slots["with_kid"] = w
    elif pending == "budget":
        b = parse_budget(text)
        if b: slots["budget"] = b

    return {"slots": slots, "pending": None}

# -----------------------------
# 3) decide_next: LLM에게 다음 액션을 JSON으로 물어보기
# -----------------------------
ALLOWED = ["ask_city", "ask_days", "ask_kid", "ask_budget", "plan"]

ROUTER_SYSTEM = SystemMessage(content=(
    "당신은 대화 오케스트레이터입니다. 사용자의 여행 정보 수집 상태를 보고 "
    "다음 단계 액션을 정확히 하나만 선택하세요. "
    "항상 JSON만 출력하세요. 추가 설명 텍스트를 넣지 마세요."
))

ROUTER_USER_TMPL = """\
[현재 슬롯 상태]
- city: {city}
- days: {days}
- with_kid: {with_kid}
- budget: {budget}

[허용 액션(정확히 하나만 선택)]
- ask_city: 도시가 비어있으면 도시를 물어본다.
- ask_days: 여행 일수가 비어있으면 일수를 물어본다.
- ask_kid: 아이 동반 여부가 비어있으면 이를 물어본다.
- ask_budget: 예산이 비어있으면 예산을 물어본다.
- plan: 모든 슬롯이 충분하면 최종 여행 일정을 생성한다.

[규칙]
1) 위 허용 액션 이름 중 하나를 action 필드로 출력.
2) 왜 그 액션을 선택했는지 reason 필드에 한 문장으로 쓰기.
3) 반드시 JSON만 출력. 예: {{"action":"ask_days","reason":"일수가 비었습니다."}}

이전 대화 맥락도 참고해서 가장 적절한 next step을 선택하세요.
"""

def _coerce_allowed(action: str | None, slots: Slots) -> AllowedAction:
    """LLM이 엉뚱한 값을 내놓았을 때의 보수적 기본값(첫 번째로 비어 있는 슬롯)"""
    if action in ALLOWED:
        return action  # type: ignore
    if not slots.get("city"): return "ask_city"
    if not slots.get("days"): return "ask_days"
    if "with_kid" not in slots: return "ask_kid"  # False일 수도 있으니 'not in' 체크
    if not slots.get("budget"): return "ask_budget"
    return "plan"

def decide_next(state: ChatState):
    slots = state.get("slots") or {}
    model_tag = state.get("model") or "llama3:8b-instruct"
    chat = ChatOllama(model=model_tag, temperature=0.0)

    user = HumanMessage(
        content=ROUTER_USER_TMPL.format(
            city=slots.get("city"),
            days=slots.get("days"),
            with_kid=slots.get("with_kid"),
            budget=slots.get("budget"),
        )
    )

    # JSON 강제: system+user 조합. (일부 모델은 JSON만 출력하도록 강하게 요구해야 함)
    resp = chat.invoke([ROUTER_SYSTEM] + state["messages"] + [user])

    raw = resp.content.strip()
    action = None
    reason = None
    try:
        data = json.loads(raw)
        action = data.get("action")
        reason = data.get("reason")
    except Exception:
        # 모델이 JSON을 지키지 못한 경우: 간단한 정규식 추출 시도
        m = re.search(r'"action"\s*:\s*"([^"]+)"', raw)
        if m:
            action = m.group(1)
        m2 = re.search(r'"reason"\s*:\s*"([^"]+)"', raw)
        if m2:
            reason = m2.group(1)

    action = _coerce_allowed(action, slots)
    if not reason:
        # 기본 reason
        reason = f"LLM route → {action}"

    # 라우팅 근거를 메시지로도 남겨 디버깅/관측에 활용
    obs = AIMessage(content=f"[router] next_action={action}, reason={reason}")
    return {"messages": [obs], "next_action": action, "reason": reason}

# -----------------------------
# 4) 질문 노드 & 최종 플랜 노드
# -----------------------------
def ask_city(state: ChatState):
    return {"messages": [AIMessage(content="어느 도시로 여행을 계획 중이신가요? 예: 시드니")],
            "pending": "city"}

def ask_days(state: ChatState):
    return {"messages": [AIMessage(content="여행 일정은 며칠로 생각하시나요? 예: 3일")],
            "pending": "days"}

def ask_kid(state: ChatState):
    return {"messages": [AIMessage(content="아이와 함께 동행하시나요? (예/아니오) 또는 아이 나이를 알려주세요. 예: 5살 아이 있어요")],
            "pending": "with_kid"}

def ask_budget(state: ChatState):
    return {"messages": [AIMessage(content="예산 수준을 알려주세요. (저렴/중간/고급) 중에서 선택해 주세요.")],
            "pending": "budget"}

def plan(state: ChatState):
    s = state.get("slots") or {}
    model_tag = state.get("model") or "llama3:8b-instruct"

    sys = SystemMessage(content=(
        "당신은 한국어 여행 플래너입니다. 답변은 '요약 → 상세 일정' 순서로 공손하게 작성하세요. "
        "아이 동반 시 유아/아동 친화 활동과 휴식 템포를 포함하고, 예산(low/mid/high)에 맞춘 팁을 제시하세요."
    ))
    user = HumanMessage(content=f"""
[요청 파라미터]
- 도시: {s.get('city')}
- 일수: {s.get('days')}일
- 아이 동반: {s.get('with_kid')}
- 예산: {s.get('budget')}

[요청]
- 하루 단위(아침/점심/오후/저녁)로 3~5개 핵심 활동
- 이동 동선 최소화
- 맛집 1~2곳/일 + 예약 팁
- 마지막에 '빠른 체크리스트'를 불릿으로 요약
""".strip())

    chat = ChatOllama(model=model_tag, temperature=0.3)
    ai = chat.invoke([sys] + state["messages"] + [user])
    return {"messages": [ai], "done": True}

# -----------------------------
# 5) 그래프 구성 (LLM 라우팅 기반)
# -----------------------------
def _read_next(state: ChatState) -> AllowedAction:
    # decide_next 노드가 설정한 next_action 을 그대로 라벨로 반환
    return state.get("next_action") or "ask_city"

def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("ingest", ingest)
    builder.add_node("decide_next", decide_next)
    builder.add_node("ask_city", ask_city)
    builder.add_node("ask_days", ask_days)
    builder.add_node("ask_kid", ask_kid)
    builder.add_node("ask_budget", ask_budget)
    builder.add_node("plan", plan)

    builder.set_entry_point("ingest")
    # ingest 이후 → decide_next 로 이동
    builder.add_edge("ingest", "decide_next")

    # decide_next 의 결과(JSON action)에 따라 분기
    builder.add_conditional_edges(
        "decide_next",
        _read_next,
        {
            "ask_city": "ask_city",
            "ask_days": "ask_days",
            "ask_kid": "ask_kid",
            "ask_budget": "ask_budget",
            "plan": "plan",
        },
    )

    # 질문 노드들은 종료(유저 응답을 기다림)
    for node in ["ask_city", "ask_days", "ask_kid", "ask_budget"]:
        builder.add_edge(node, END)

    builder.add_edge("plan", END)

    # 체크포인터(메모리): 멀티턴 유지
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)

# -----------------------------
# 6) 데모(4턴 대화)
# -----------------------------
if __name__ == "__main__":
    graph = build_graph()
    thread_id = "trip-llm-routing-demo"
    cfg = {"configurable": {"thread_id": thread_id}}

    def turn(user_text: str, model_tag: str = "llama3:8b-instruct"):
        res = graph.invoke(
            {
                "messages": [HumanMessage(content=user_text)],
                "model": model_tag,
                "slots": {},
                "pending": None,
                "next_action": None,
                "reason": None,
                "done": False,
            },
            config=cfg
        )
        # 마지막 AI 메시지 출력
        ai = next((m for m in reversed(res["messages"]) if isinstance(m, AIMessage)), None)
        print("\n[Assistant]\n", ai.content if ai else "(응답 없음)")
        return res

    print("=== 턴 1 ===")
    turn("7월에 가족이랑 3일 일정으로 호주 여행 가고 싶어요. 추천해줘.", "llama3:8b-instruct")

    print("\n=== 턴 2 ===")
    turn("시드니요")

    print("\n=== 턴 3 ===")
    turn("5살 아이 있어요")

    print("\n=== 턴 4 ===")
    turn("중간 정도요")  # 여기서 plan 액션으로 분기되어 최종 일정 출력 예상

    # (선택) Qwen으로 동일 흐름을 별도 스레드에서 시험
    print("\n=== Qwen 스레드 ===")
    cfg_q = {"configurable": {"thread_id": "trip-llm-routing-qwen"}}
    res = graph.invoke(
        {
            "messages": [HumanMessage(content="가족 2일 도쿄, 저렴하게 부탁해요.")],
            "model": "qwen3:8b-instruct",  # 없으면 qwen2.5:7b/14b-instruct
            "slots": {},
            "pending": None,
            "next_action": None,
            "reason": None,
            "done": False,
        },
        config=cfg_q
    )
    ai = next((m for m in reversed(res["messages"]) if isinstance(m, AIMessage)), None)
    print("\n[Assistant/Qwen]\n", ai.content if ai else "(응답 없음)")
```

---

## 동작 설명

1. **LLM 라우팅(decide\_next)**

   * `ROUTER_SYSTEM` + `ROUTER_USER_TMPL`로 **오직 JSON 형태**(`{"action": "...", "reason": "..."}`) 출력을 요구합니다.
   * 모델이 JSON을 어기면 정규식으로 보정 → 그래도 실패하면 `_coerce_allowed()`로 **보수적 기본값**(첫 번째 비어있는 슬롯) 선택.

2. **조건부 에지 연결**

   * `decide_next` 노드가 `state["next_action"]`을 세팅하면,
     `add_conditional_edges("decide_next", _read_next, {...})`가 **그 값을 라벨로 사용**하여 해당 노드로 분기합니다.

3. **멀티턴 루프**

   * `ask_*` 노드는 질문만 남기고 `END` → 다음 턴에 사용자가 응답 → `ingest`가 슬롯 흡수 → `decide_next`가 다시 판단 → … → `plan`.

4. **모델 독립성**

   * `state["model"]` 문자열만 바꾸면 **Llama ↔ Qwen** 교체가 바로 됩니다.
   * `qwen3:8b-instruct` 태그가 환경에 없을 수 있으므로, **가용한 Qwen instruct 태그**로 대체하셔도 흐름은 동일합니다.

---

## 확장 포인트

* **엄격한 구조화 출력**:
  Pydantic/JSON 스키마를 프롬프트에 포함하거나, “항상 code-fence 없이 JSON” 규칙을 더 강하게 명시해 일관성 향상.
* **전용 라우터 모델**:
  분기 판단은 경량 모델(예: 3\~7B)로, 본문 생성은 더 큰 모델로 **이원화** 가능(`router_model`/`generator_model` 분리).
* **관측/로깅**:
  `reason`과 `[router] ...` 메시지를 APM/로그로 수집해 **왜 그 분기가 일어났는지** 추적.
* **체크포인터 교체**:
  `MemorySaver` → Redis/PostgreSQL 체크포인터로 바꾸면 **영속 멀티턴**과 **수평 확장**이 쉬워집니다.
* **툴 호출 결합**:
  `plan` 전에 `weather_api`, `poi_search`, `price_lookup` 등의 **툴 노드**를 배치하고, LLM 라우터가 *툴 호출 여부*까지 결정하도록 확장할 수 있습니다.
* **스트리밍 응답**:
  `graph.stream(inputs, stream_mode="values")`로 토큰 스트리밍 → SSE/WebSocket으로 프론트에 전달.

---

원하시면 위 예제를 **FastAPI 엔드포인트**로 감싸서 `thread_id` 기준 멀티턴 세션을 유지하는 서버 버전, 또는 **라우터 전용 소형 LLM + 본문 생성 대형 LLM** 이원화 템플릿도 바로 제공해 드리겠습니다.

---
---
---

# 체크포인트, 서브그래프, 스트리밍

아래 답변은 **요약 → 각 케이스별 코드(최소 2개씩) → 실행/확장 가이드** 순서로 구성했습니다.
요청하신 대로 **LangGraph의 체크포인트 / 서브그래프 / 다양한 스트리밍 모드 / 에러 복구**를 모두 다루고, **각 항목당 최소 2개 이상의 예제 코드**를 제공합니다. 모든 예제는 **Ollama + Llama3/Qwen** 환경을 가정합니다.

---

## 요약

* **공통 전제**

  * Python ≥ 3.10 권장
  * `pip install -U langgraph langchain langchain-community ollama fastapi sse-starlette uvicorn`
  * Ollama 모델(예):

    * `ollama pull llama3:8b-instruct`
    * `ollama pull qwen3:8b-instruct` *(환경에 없으면 `qwen2.5:7b/14b-instruct`로 대체)*
  * `ollama serve` 실행 중
* **체크포인트**:

  * **Case 1A**: `MemorySaver` 기반 멀티턴
  * **Case 1B**: `SqliteSaver` 기반 영속화(프로세스 재시작에도 유지)
* **서브그래프**:

  * **Case 2A**: “요구사항 수집(Collect)”를 별도 서브그래프로 분리
  * **Case 2B**: “라우터 서브그래프(LLM 판단) → 생성 서브그래프(플랜 작성)” 다단계 중첩
* **스트리밍 모드**:

  * **Case 3A**: `graph.stream(..., stream_mode="values")` 값 스트림
  * **Case 3B**: `graph.stream(..., stream_mode="updates")` 노드별 업데이트 스트림 + **SSE** 서버
* **에러 복구**:

  * **Case 4A**: 노드 내부 `try/except` + 상태 플래그로 회복 경로 진입
  * **Case 4B**: 실패 시 **재시도/폴백 모델 전환**(리트라이 카운트/백오프)
* **조합**:

  * **Case 5A**: 체크포인트(영속) + 서브그래프 + SSE 스트리밍
  * **Case 5B**: 체크포인트 + 스트리밍(updates) + 폴백 리커버리(재시도)

> 아래 코드는 파일 단위로 분리해도 되고, 한 프로젝트 내에 함께 두고 실행하셔도 됩니다.

---

# 공통 유틸(선택): 메시지/상태 타입

아래 예제 대부분에서 동일 타입을 사용합니다(필수 아님).

```python
# file: common_types.py
from __future__ import annotations
from typing import Annotated, TypedDict, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

AllowedAction = Literal["ask_city", "ask_days", "ask_kid", "ask_budget", "plan"]

class Slots(TypedDict, total=False):
    city: str
    days: int
    with_kid: bool
    budget: Literal["low", "mid", "high"]

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    slots: Slots
    pending: str | None
    model: str
    next_action: AllowedAction | None
    reason: str | None
    done: bool
    error: bool | None
    retry: int | None
```

---

## ✅ 체크포인트(Checkpoint) — 2가지 케이스

### Case 1A) `MemorySaver` 기반 멀티턴(가벼운 세션 유지)

```python
# file: cp_mem_basic.py
from __future__ import annotations
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from common_types import ChatState

def llm_node(state: ChatState):
    chat = ChatOllama(model=state.get("model") or "llama3:8b-instruct", temperature=0.3)
    ai = chat.invoke(state["messages"])
    return {"messages": [ai]}

def build_graph():
    sg = StateGraph(ChatState)
    sg.add_node("chat", llm_node)
    sg.set_entry_point("chat")
    sg.add_edge("chat", END)
    return sg.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "mem-demo-1"}}
    # 턴 1
    res = graph.invoke({"messages":[HumanMessage("안녕하세요! 여행 도와줄 수 있나요?")], "model":"llama3:8b-instruct",
                        "slots":{}, "pending":None, "next_action":None, "reason":None, "done":False}, config=cfg)
    print(res["messages"][-1].content)
    # 턴 2(이전 대화 자동 유지)
    res = graph.invoke({"messages":[HumanMessage("시드니 3일 일정 권해줘")]}, config=cfg)
    print(res["messages"][-1].content)
```

### Case 1B) `SqliteSaver` 기반 영속 체크포인트(재시작에도 유지)

```python
# file: cp_sqlite_persistent.py
from __future__ import annotations
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
from common_types import ChatState

def llm_node(state: ChatState):
    chat = ChatOllama(model=state.get("model") or "llama3:8b-instruct", temperature=0.2)
    ai = chat.invoke(state["messages"])
    return {"messages":[ai]}

def build_graph(sqlite_path="checkpoints.sqlite"):
    conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    sg = StateGraph(ChatState)
    sg.add_node("chat", llm_node)
    sg.set_entry_point("chat")
    sg.add_edge("chat", END)
    return sg.compile(checkpointer=saver)

if __name__ == "__main__":
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "sqlite-demo-1"}}
    # 프로세스 A에서
    graph.invoke({"messages":[HumanMessage("첫 대화: 도시 추천해줘")], "model":"llama3:8b-instruct",
                  "slots":{}, "pending":None, "next_action":None, "reason":None, "done":False}, config=cfg)
    # 프로세스 재시작/다른 스크립트에서도 cfg 동일하면 맥락 이어짐
```

---

## ✅ 서브그래프(Subgraph) — 2가지 케이스

### Case 2A) “요구사항 수집(Collect)”을 서브그래프로 분리

```python
# file: subgraph_collect.py
from __future__ import annotations
from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from common_types import ChatState

def ask_city(state: ChatState):   return {"messages":[AIMessage("어느 도시로 가시나요?")], "pending":"city"}
def ask_days(state: ChatState):   return {"messages":[AIMessage("며칠 일정인가요? (예: 3일)")], "pending":"days"}
def collect_route(state: ChatState) -> Literal["city","days","done"]:
    s = state.get("slots") or {}
    if not s.get("city"): return "city"
    if not s.get("days"): return "days"
    return "done"

def ingest(state: ChatState):
    # 사용자 최신 입력을 pending 슬롯에 반영(간단화)
    last = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last or not state.get("pending"): return {}
    s = dict(state.get("slots") or {})
    if state["pending"]=="city": s["city"]=last.content.strip()
    if state["pending"]=="days":
        import re
        m=re.search(r"(\d+)\s*일",last.content)
        if m: s["days"]=int(m.group(1))
    return {"slots":s, "pending":None}

def build_collect_subgraph():
    g = StateGraph(ChatState)
    g.add_node("ingest", ingest)
    g.add_node("ask_city", ask_city)
    g.add_node("ask_days", ask_days)
    g.set_entry_point("ingest")
    g.add_conditional_edges("ingest", collect_route,
        {"city":"ask_city", "days":"ask_days", "done":END})
    g.add_edge("ask_city", END)
    g.add_edge("ask_days", END)
    return g.compile()

# 상위 그래프: collect 서브그래프 실행 후 다음 단계로
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama

def plan(state: ChatState):
    from langchain_core.messages import SystemMessage
    sys = SystemMessage("한국어로 공손히, 요약→상세 순")
    chat = ChatOllama(model=state.get("model") or "llama3:8b-instruct")
    ai = chat.invoke([sys]+state["messages"]+[HumanMessage(f"slots={state.get('slots')} 기반 일정 제안")])
    return {"messages":[ai], "done":True}

def build_parent():
    sg = StateGraph(ChatState)
    sg.add_node("collect", build_collect_subgraph())  # 서브그래프를 노드로 추가
    sg.add_node("plan", plan)
    sg.set_entry_point("collect")
    sg.add_edge("collect", "plan")
    sg.add_edge("plan", END)
    return sg.compile()

if __name__=="__main__":
    graph = build_parent()
    tid = {"configurable":{"thread_id":"sub-collect"}}
    # 턴1 → 도시 질문
    res = graph.invoke({"messages":[HumanMessage("호주 3일 일정 부탁해요")], "model":"llama3:8b-instruct",
                        "slots":{}, "pending":None, "done":False}, config=tid)
    print(res["messages"][-1].content)
    # 턴2 → 사용자 답
    res = graph.invoke({"messages":[HumanMessage("시드니")],}, config=tid)
    print(res["messages"][-1].content)
    # 턴3 → 사용자 답
    res = graph.invoke({"messages":[HumanMessage("3일")],}, config=tid)
    print(res["messages"][-1].content)  # 여기서 계획 생성 답변
```

### Case 2B) “라우터(LLM 판단) 서브그래프 → 생성 서브그래프” 구성

```python
# file: subgraph_router_then_generate.py
from __future__ import annotations
import json, re
from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from common_types import ChatState, AllowedAction

# --- 라우터 서브그래프(LLM이 next_action JSON으로 출력) ---
def decide_next(state: ChatState):
    tmpl = """{"action":"{a}","reason":"{r}"}"""
    s = state.get("slots") or {}
    # 간단 프롬프트
    sys = SystemMessage("JSON으로만 출력하세요. action ∈ {ask_city,ask_days,plan}")
    user = HumanMessage(f"slots={s} → 다음 액션 추천")
    chat = ChatOllama(model=state.get("model") or "llama3:8b-instruct", temperature=0.0)
    resp = chat.invoke([sys]+state["messages"]+[user]).content.strip()
    try: action = json.loads(resp).get("action")
    except: 
        m=re.search(r'"action"\s*:\s*"([^"]+)"',resp); action=m.group(1) if m else "ask_city"
    if action not in {"ask_city","ask_days","plan"}: action="ask_city"
    return {"next_action": action, "messages":[AIMessage(f"[router] action={action}")]}

def ask_city(state: ChatState): return {"messages":[AIMessage("도시는 어디인가요?")], "pending":"city"}
def ask_days(state: ChatState): return {"messages":[AIMessage("며칠 계획인가요? (예: 3일)")], "pending":"days"}

def route_label(state: ChatState) -> Literal["city","days","plan"]:
    a = state.get("next_action") or "city"
    return "city" if a=="ask_city" else ("days" if a=="ask_days" else "plan")

def build_router_subgraph():
    g = StateGraph(ChatState)
    g.add_node("decide", decide_next)
    g.add_node("ask_city", ask_city)
    g.add_node("ask_days", ask_days)
    g.set_entry_point("decide")
    g.add_conditional_edges("decide", route_label, {"city":"ask_city","days":"ask_days","plan":END})
    g.add_edge("ask_city", END)
    g.add_edge("ask_days", END)
    return g.compile()

# --- 생성 서브그래프(최종 답변) ---
def generate_plan(state: ChatState):
    chat = ChatOllama(model=state.get("model") or "qwen3:8b-instruct", temperature=0.3)
    ai = chat.invoke([HumanMessage(f"slots={state.get('slots')}로 가족 친화 일정")] )
    return {"messages":[ai], "done":True}

def parent_build():
    g = StateGraph(ChatState)
    g.add_node("router", build_router_subgraph())
    g.add_node("generate", generate_plan)
    g.set_entry_point("router")
    g.add_edge("router","generate")
    g.add_edge("generate", END)
    return g.compile()

if __name__=="__main__":
    graph = parent_build()
    cfg={"configurable":{"thread_id":"router-gen"}}
    res = graph.invoke({"messages":[HumanMessage("가족 3일 일정 추천")], "model":"llama3:8b-instruct",
                        "slots":{}, "pending":None, "done":False}, config=cfg)
    print(res["messages"][-1].content)
```

---

## ✅ 스트리밍(Streaming) — 2가지 케이스

### Case 3A) `stream_mode="values"`: 단계별 “상태 값” 스트리밍

```python
# file: stream_values.py
from __future__ import annotations
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
from common_types import ChatState
from langgraph.checkpoint.memory import MemorySaver

def llm_node(state: ChatState):
    chat = ChatOllama(model=state.get("model") or "llama3:8b-instruct", temperature=0.2)
    ai = chat.invoke(state["messages"])
    return {"messages":[ai]}

def build_graph():
    g = StateGraph(ChatState)
    g.add_node("llm", llm_node)
    g.set_entry_point("llm")
    g.add_edge("llm", END)
    return g.compile(checkpointer=MemorySaver())

if __name__=="__main__":
    graph = build_graph()
    cfg={"configurable":{"thread_id":"stream-values"}}
    inputs = {"messages":[HumanMessage("요약→상세로 시드니 3일 일정")]}
    for chunk in graph.stream(inputs, config=cfg, stream_mode="values"):
        # chunk는 누적 상태의 스냅샷(사전)입니다.
        if "messages" in chunk and chunk["messages"]:
            last = chunk["messages"][-1]
            print(">> partial:", getattr(last, "content", "")[:120], "...")
```

### Case 3B) `stream_mode="updates"` + **SSE** 서버(노드별 업데이트 스트림)

```python
# file: stream_updates_sse.py
from __future__ import annotations
from typing import Any, Dict
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from common_types import ChatState

def step1(state: ChatState):
    return {"messages":[AIMessage("요구사항을 파악하겠습니다. 도시/일수 알려주세요.")]}
def step2(state: ChatState):
    chat = ChatOllama(model=state.get("model") or "llama3:8b-instruct")
    ai = chat.invoke(state["messages"])
    return {"messages":[ai]}

def build_graph():
    g = StateGraph(ChatState)
    g.add_node("step1", step1)
    g.add_node("step2", step2)
    g.set_entry_point("step1")
    g.add_edge("step1","step2")
    g.add_edge("step2", END)
    return g.compile(checkpointer=MemorySaver())

app = FastAPI()
graph = build_graph()

@app.get("/sse")
async def sse(prompt: str):
    cfg={"configurable":{"thread_id":"sse-demo"}}
    inputs={"messages":[HumanMessage(prompt)], "model":"llama3:8b-instruct"}
    def gen():
        for upd in graph.stream(inputs, config=cfg, stream_mode="updates"):
            # upd: {"node_name": {"keys_changed": ... , ...}}
            for node_name, payload in upd.items():
                yield {"event":"update", "data": f"{node_name}:{list(payload.keys())}"}
        yield {"event":"done","data":"[DONE]"}
    return EventSourceResponse(gen())

# 실행: uvicorn stream_updates_sse:app --reload
```

---

## ✅ 에러 복구(Recovery) — 2가지 케이스

### Case 4A) 노드 내부 `try/except` + 리커버리 경로

```python
# file: recover_node_try_except.py
from __future__ import annotations
import random
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from common_types import ChatState

def risky_tool(state: ChatState):
    # 실패를 흉내내는 코드
    if any("fail" in getattr(m,"content","").lower() for m in state["messages"]):
        raise RuntimeError("외부 API 오류")
    if random.random() < 0.2:
        raise RuntimeError("일시 장애(랜덤)")
    return {"messages":[AIMessage("도구 호출 성공: 결과 요약...")]}

def node_with_recovery(state: ChatState):
    try:
        return risky_tool(state)
    except Exception as e:
        # 오류 상태로 마킹하고 복구 노드로 라우팅
        return {"messages":[AIMessage(f"[오류] {e}. 대안 경로로 전환합니다.")], "error": True}

def recover(state: ChatState):
    # 폴백: 모델에게 직접 생성하게 하거나 캐시/샘플로 대체
    chat = ChatOllama(model=state.get("model") or "qwen3:8b-instruct", temperature=0.1)
    ai = chat.invoke(state["messages"] + [HumanMessage("도구 없음. 최대한 유용한 대안을 제시해줘.")])
    return {"messages":[ai], "error": False}

def route(state: ChatState) -> Literal["ok","recover"]:
    return "recover" if state.get("error") else "ok"

def build_graph():
    g = StateGraph(ChatState)
    g.add_node("step", node_with_recovery)
    g.add_node("recover", recover)
    g.set_entry_point("step")
    g.add_conditional_edges("step", route, {"ok": END, "recover": "recover"})
    g.add_edge("recover", END)
    return g.compile(checkpointer=MemorySaver())

if __name__=="__main__":
    graph = build_graph()
    cfg={"configurable":{"thread_id":"recover-try"}}
    # 실패 유도: "fail" 포함
    res = graph.invoke({"messages":[HumanMessage("이 요청은 fail 을 포함합니다.")], "model":"llama3:8b-instruct"}, config=cfg)
    print(res["messages"][-1].content)
```

### Case 4B) 재시도/백오프 + 폴백 모델 전환

```python
# file: recover_retry_fallback.py
from __future__ import annotations
import time
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from common_types import ChatState

MAX_RETRY = 2

def gen_with_retry(state: ChatState):
    retry = int(state.get("retry") or 0)
    model = state.get("model") or "llama3:8b-instruct"
    try:
        chat = ChatOllama(model=model, temperature=0.2)
        ai = chat.invoke(state["messages"])
        return {"messages":[ai], "error": False, "retry": retry}
    except Exception as e:
        if retry < MAX_RETRY:
            time.sleep(0.5 * (retry+1))  # 간단 백오프
            return {"messages":[AIMessage(f"[경고] 모델 {model} 실패, 재시도={retry+1}")],
                    "error": True, "retry": retry+1}
        # 폴백 모델로 전환
        fallback = "qwen3:8b-instruct"
        chat = ChatOllama(model=fallback, temperature=0.2)
        ai = chat.invoke(state["messages"]+[HumanMessage(f"메인 모델 실패. 폴백({fallback})로 처리")])
        return {"messages":[ai], "error": False, "retry": retry}

def route(state: ChatState) -> Literal["retry","done"]:
    return "retry" if state.get("error") else "done"

def build_graph():
    g = StateGraph(ChatState)
    g.add_node("gen", gen_with_retry)
    g.set_entry_point("gen")
    g.add_conditional_edges("gen", route, {"retry":"gen","done":END})
    return g.compile(checkpointer=MemorySaver())

if __name__=="__main__":
    graph = build_graph()
    cfg={"configurable":{"thread_id":"retry-fallback"}}
    res = graph.invoke({"messages":[HumanMessage("네트워크가 불안정한 상황을 가정하고 답변해줘.")],
                        "model":"llama3:8b-instruct", "retry":0}, config=cfg)
    print(res["messages"][-1].content)
```

---

## ✅ 조합(Combinations) — 2가지 케이스

### Case 5A) **체크포인트(영속) + 서브그래프 + SSE 스트리밍(values)**

```python
# file: combo_cp_subgraph_sse.py
from __future__ import annotations
import sqlite3
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from common_types import ChatState

def collect_node(state: ChatState):
    # 아주 간단한 수집(실전은 subgraph_collect.py 재사용 권장)
    s = dict(state.get("slots") or {})
    last = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if last and "시드니" in last.content:
        s["city"]="시드니"
    return {"slots": s, "messages":[AIMessage("요구사항 수집 완료(간소)")]}

def plan_node(state: ChatState):
    chat = ChatOllama(model=state.get("model") or "llama3:8b-instruct")
    ai = chat.invoke(state["messages"]+[HumanMessage(f"slots={state.get('slots')} 기반 일정")])
    return {"messages":[ai], "done":True}

def build_graph(db="combo.sqlite"):
    conn = sqlite3.connect(db, check_same_thread=False)
    saver = SqliteSaver(conn)
    g = StateGraph(ChatState)
    g.add_node("collect", collect_node)
    g.add_node("plan", plan_node)
    g.set_entry_point("collect")
    g.add_edge("collect","plan")
    g.add_edge("plan", END)
    return g.compile(checkpointer=saver)

app = FastAPI()
graph = build_graph()

@app.get("/values")
async def values(prompt: str, thread_id: str = "combo-1"):
    cfg={"configurable":{"thread_id":thread_id}}
    inputs={"messages":[HumanMessage(prompt)], "model":"llama3:8b-instruct"}
    def gen():
        for chunk in graph.stream(inputs, config=cfg, stream_mode="values"):
            yield {"event":"value","data":str(list(chunk.keys()))}
        yield {"event":"done","data":"[DONE]"}
    return EventSourceResponse(gen())

# 실행: uvicorn combo_cp_subgraph_sse:app --reload
```

### Case 5B) **체크포인트 + 스트리밍(updates) + 폴백 리커버리(재시도)**

```python
# file: combo_updates_recover.py
from __future__ import annotations
import time
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from common_types import ChatState

MAX_RETRY=1

def risky_gen(state: ChatState):
    model = state.get("model") or "llama3:8b-instruct"
    retry = int(state.get("retry") or 0)
    try:
        chat = ChatOllama(model=model, temperature=0.2)
        ai = chat.invoke(state["messages"])
        return {"messages":[ai], "error":False, "retry":retry}
    except Exception as e:
        if retry < MAX_RETRY:
            return {"messages":[AIMessage(f"[경고] {e}. 재시도={retry+1}")], "error":True, "retry":retry+1}
        fb = "qwen3:8b-instruct"
        chat = ChatOllama(model=fb, temperature=0.2)
        ai = chat.invoke(state["messages"]+[HumanMessage(f"폴백 {fb} 사용")])
        return {"messages":[ai], "error":False, "retry":retry}

def route(state: ChatState): return "retry" if state.get("error") else "done"

def build_graph():
    g = StateGraph(ChatState)
    g.add_node("gen", risky_gen)
    g.set_entry_point("gen")
    g.add_conditional_edges("gen", route, {"retry":"gen","done":END})
    return g.compile(checkpointer=MemorySaver())

app = FastAPI()
graph = build_graph()

@app.get("/updates")
async def updates(prompt: str, thread_id: str = "combo-2"):
    cfg={"configurable":{"thread_id":thread_id}}
    inputs={"messages":[HumanMessage(prompt)], "model":"llama3:8b-instruct", "retry":0}
    def gen():
        for upd in graph.stream(inputs, config=cfg, stream_mode="updates"):
            for node, payload in upd.items():
                yield {"event":"update", "data": f"{node}:{list(payload.keys())}"}
        yield {"event":"done","data":"[DONE]"}
    return EventSourceResponse(gen())

# 실행: uvicorn combo_updates_recover:app --reload
```

---

## 실행/확장 가이드

1. **설치 및 모델 준비**

   ```bash
   pip install -U langgraph langchain langchain-community ollama fastapi sse-starlette uvicorn
   ollama pull llama3:8b-instruct
   ollama pull qwen3:8b-instruct     # 가용X면 qwen2.5:7b/14b-instruct
   ollama serve
   ```

2. **실행 팁**

   * 스트리밍 서버 예제:
     `uvicorn stream_updates_sse:app --reload` 후 브라우저에서 `/sse?prompt=...` 또는 `/values?prompt=...` 접근
   * Sqlite 체크포인트: `checkpoints.sqlite`/`combo.sqlite` 파일이 생성되며, **프로세스 재시작 후에도 스레드 맥락 유지**
   * 멀티 세션: `thread_id` 파라미터로 동시 세션 분리

3. **확장 포인트**

   * **체크포인터**: MemorySaver → `SqliteSaver` → Redis/Postgres로 변경(운영 환경 영속/동시성)
   * **서브그래프 계층화**: 수집/검증/툴호출/생성 등 도메인별 서브그래프를 **모듈화**
   * **스트리밍 UX**: SSE 외에 WebSocket/Server Push 적용, 프런트에서 토큰/스텝 구분 렌더
   * **에러 정책**: 재시도 횟수, 백오프, 폴백 모델/툴, “부분 성공(Partial)” 응답 설계, 사용자에게 복구 질문(예산 상향/요건완화 등)

---

원하시면 위 코드를 **하나의 예제 레포** 구조로 묶고, **Dockerfile + docker-compose**(Ollama/앱 동시 기동), **Redis/PostgreSQL 체크포인터** 버전, 그리고 **라우터 전용 소형 LLM + 생성 전용 LLM** 이원화 템플릿까지 확장한 패키지를 추가로 드리겠습니다.

---
---
---

# 통합 및 에러복구

아래는 **요약 → 실행 방법 → 레포 구조 → 주요 코드 설명(이원화 라우터/생성 포함) → 확장/운영 팁** 순서입니다.
요청하신 1) 예제 레포 구조, 2) Dockerfile + docker-compose(오라마+앱 동시 기동), 3) **라우터 전용 소형 LLM + 생성 전용 LLM** 이원화 템플릿을 모두 포함한 **완성 ZIP**을 바로 제공합니다.

# 요약

* LangGraph 멀티턴 예제들을 **하나의 레포**로 묶었고, 체크포인트(Memory/SQLite), 서브그래프, **두 가지 스트리밍 모드(values/updates)**, 에러 복구(try/except 리커버리 · 재시도/폴백), **LLM 라우팅 + 생성 이원화** 템플릿까지 포함했습니다.
* Docker Compose로 **Ollama와 FastAPI 앱을 함께 띄우는** 구성을 제공합니다.
* 모델 태그는 환경 변수를 통해 손쉽게 교체할 수 있습니다. (기본: `ROUTER_MODEL=qwen2.5:3b-instruct`, `GENERATOR_MODEL=llama3:8b-instruct`)

**다운로드:** [langgraph-ollama-multiturn-demo.zip](sandbox:/mnt/data/langgraph-ollama-multiturn-demo.zip)

---

# 빠른 실행 방법

1. Docker로 실행 (권장)

```bash
# 압축 해제 후 프로젝트 루트에서
cp .env.example .env        # 필요 시 모델 태그 수정
docker compose up --build
# 앱: http://localhost:8000  (OpenAPI 문서: /docs)
```

2. 로컬 파이썬으로 실행 (Ollama는 별도로 구동)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OLLAMA_HOST=http://localhost:11434
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> **모델 태그 주의사항**: 환경/시점에 따라 `qwen3:8b-instruct`가 없을 수 있습니다. 이 경우 `qwen2.5:7b-instruct` 또는 `qwen2.5:14b-instruct` 등 **가용한 instruct 태그**로 바꿔 사용하시면 흐름은 동일하게 동작합니다.

---

# 레포 구조

```
langgraph-ollama-multiturn-demo/
├─ README.md
├─ requirements.txt
├─ .env.example                 # ROUTER_MODEL / GENERATOR_MODEL / DEFAULT_MODEL / OLLAMA_HOST
├─ Dockerfile                   # FastAPI 앱 컨테이너
├─ docker-compose.yml           # ollama + app 동시 기동
├─ scripts/
│  └─ pull_models.sh            # (옵션) 모델 프리풀
└─ app/
   ├─ main.py                   # FastAPI 엔드포인트 (/chat, /sse/values, /sse/updates)
   ├─ common_types.py           # 공통 State/TypedDict
   ├─ utils/
   │  └─ model_tags.py          # 환경변수 기반 모델 태그/OLLAMA_HOST
   └─ graphs/
      ├─ cp_mem_basic.py        # Checkpoint: MemorySaver
      ├─ cp_sqlite_persistent.py# Checkpoint: SqliteSaver (/data/checkpoints.sqlite)
      ├─ subgraph_collect.py    # Subgraph: 수집 → 최종 plan
      ├─ subgraph_router_then_generate.py   # LLM Router(서브그래프) → Generator(서브그래프)
      ├─ multiturn_llm_branching.py         # LLM JSON 라우팅 + 슬롯필링 + 체크포인트
      ├─ recover_node_try_except.py         # try/except 리커버리 플로우
      └─ recover_retry_fallback.py          # 재시도 + 폴백 모델 전환
```

---

# 주요 사용법

## 1) 단발 대화 호출

```
POST /chat
{
  "prompt": "가족 3일 시드니 일정 추천",
  "graph": "llm_router_multiturn",   // 아래 목록 중 선택
  "thread_id": "t1",
  "model": "llama3:8b-instruct"      // 옵션
}
```

**그래프 이름 목록**

* `basic` : MemorySaver + 단일 노드
* `sqlite` : SqliteSaver + 단일 노드
* `collect_plan` : 수집(서브그래프) → 플랜
* `router_generate` : **라우터(서브그래프, LLM 판단/소형) → 생성(서브그래프/대형)**
* `llm_router_multiturn` : **LLM JSON 라우팅 + 멀티턴 슬롯필링 + 체크포인트** + **이원화 모델(라우터/생성)**
* `recover_try` : try/except 리커버리 경로
* `retry_fallback` : 재시도 + 폴백 모델

## 2) 스트리밍 (SSE)

* values 스트림: `GET /sse/values?graph=collect_plan&prompt=...&thread_id=...`
* updates 스트림: `GET /sse/updates?graph=router_generate&prompt=...&thread_id=...`

프런트에서 예시:

```js
const es = new EventSource("/sse/updates?graph=router_generate&prompt=Hello&thread_id=t1");
es.addEventListener("update", (e) => console.log("update:", e.data));
es.addEventListener("done",   () => es.close());
```

---

# 이원화 템플릿(라우터/생성 모델 분리) 핵심

* `app/utils/model_tags.py`에서 환경변수로 **분리 설정**

  * `ROUTER_MODEL` (기본 `qwen2.5:3b-instruct`) → **분기 판단 전용 경량 LLM**
  * `GENERATOR_MODEL` (기본 `llama3:8b-instruct`) → **실제 응답/플랜 생성 전용**
* `subgraph_router_then_generate.py` / `multiturn_llm_branching.py` 에서:

  * **Router 단계**는 `ChatOllama(model=ROUTER_MODEL, temperature=0.0)`로 JSON만 출력하게 강제
  * **Generate 단계**는 `ChatOllama(model=GENERATOR_MODEL, temperature≈0.3)`로 본문 생성

이렇게 분리하면 **비용/지연**을 줄이면서 **정확한 분기**와 **풍부한 출력**을 동시에 달성할 수 있습니다.

---

# Docker 구성 설명

* `docker-compose.yml`

  * `ollama` 서비스: 11434 포트 노출, 볼륨 `ollama:`에 모델 캐시
  * `app` 서비스: FastAPI 앱. 환경변수로 모델 태그 지정, `/data` 볼륨(`app_data`)에 SQLite 체크포인트 영속화
  * 헬스체크로 ollama 준비 후 앱 기동

* `Dockerfile`

  * `requirements.txt` 설치 → `app/` 복사 → `uvicorn`으로 8000 포트 서비스
  * 기본 환경변수(모델 태그/OLLAMA\_HOST)는 `docker-compose`에서 덮어쓰게 되어 있어 환경 맞춤 간편

---

# 에러 복구/재시도 포인트

* `recover_node_try_except.py`

  * 외부 도구 호출 실패를 시뮬레이션 → **오류 플래그로 라우팅** → **폴백 생성 경로**로 회복
* `recover_retry_fallback.py`

  * **재시도(백오프)** 후에도 실패 시 **폴백 모델로 자동 전환**
  * 운영 시에는 재시도 횟수/백오프/알림(웹훅) 등을 환경변수화 가능

---

# 운영/확장 팁

1. **체크포인터**

   * MemorySaver → `cp_sqlite_persistent.py` 처럼 `SqliteSaver` → Redis/PostgreSQL로 확장
   * 컨테이너 볼륨(`/data`)로 영속화하여 재배포에도 세션 유지

2. **서브그래프 모듈화**

   * 수집/검증/툴호출/생성/결제 등 도메인 기능별로 서브그래프를 분리하여 재사용

3. **스트리밍 UX**

   * values: 상태 스냅샷 / updates: 노드별 변경 키
   * 토큰 단위 스트리밍이 필요하면 모델 레벨 `stream()` 조합으로 확장 가능

4. **모델 가용성**

   * `scripts/pull_models.sh` 로 미리 `ollama pull` 시도
   * 라우터는 경량(예: `qwen2.5:3b-instruct`), 생성은 7B/8B 이상의 instruct 권장

---

필요하시면 **WebSocket 버전 스트리밍**, **Redis/PostgreSQL 체크포인터**, **툴 호출(날씨/POI/가격) 서브그래프** 추가 템플릿도 이어서 만들어 드리겠습니다.

---
---
---

# RAG

아래는 **요약 → 실행 방법 → 각 RAG 유형별 설계/코드 안내 → 확장 팁** 순서입니다.
요청하신 **Basic RAG / HQ-RAG / GraphRAG** 예제를 기존 레포에 통합해 두었습니다. 바로 받아 실행하실 수 있도록 **완성 ZIP**도 제공합니다.

# 요약

* 기존 LangGraph + Ollama 예제 레포에 **RAG 3종**을 추가했습니다.

  * **Basic RAG**: FAISS + Ollama 임베딩(`nomic-embed-text`) + 단순 상위 k 조합.
  * **HQ-RAG**: 멀티쿼리 확장(MQE) → RRF 융합 → **LLM 재랭킹** → 생성. (라우터/소형 모델을 재활용해 재랭킹 비용 절감 가능)
  * **GraphRAG**: LLM으로 **엔티티/관계 추출 → 그래프(NetworkX) 구축** → 질의 시 **시드 엔티티 추출 → hop 기반 경로 수집 → 근거 생성**.
* FastAPI 엔드포인트:

  * `POST /rag/basic`, `POST /rag/hq`, `POST /rag/graph`
* 인덱싱 스크립트:

  * `python -m app.rag.index_basic` (FAISS)
  * `python -m app.rag.index_graph` (GraphRAG용 그래프 JSON)
* 샘플 문서 2개를 넣어 간단 동작 확인 가능.

**다운로드:** [langgraph-ollama-multiturn-demo-rag.zip](sandbox:/mnt/data/langgraph-ollama-multiturn-demo-rag.zip)

---

# 실행 방법

## 1) Docker (권장)

```bash
# 압축 해제 후 프로젝트 루트
cp .env.example .env
# 필요 시 모델 태그/임베딩 모델 수정 (기본 EMBED_MODEL은 nomic-embed-text)
docker compose up --build
```

임베딩 모델 프리풀(선택):

```bash
./scripts/pull_models.sh
# 또는 컨테이너 내부에서: ollama pull nomic-embed-text
```

## 2) 로컬 파이썬

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OLLAMA_HOST=http://localhost:11434
ollama pull nomic-embed-text
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 3) 인덱싱

```bash
# Basic/HQ 공용 FAISS 인덱스
python -m app.rag.index_basic --docs /data/docs --out /data/index/faiss_basic
python -m app.rag.index_basic --docs /data/docs --out /data/index/faiss_hq

# GraphRAG 그래프 구축
python -m app.rag.index_graph  --docs /data/docs --out /data/index/graph.json
```

> 기본 샘플 문서 위치: `data/docs/`
> 인덱스 저장 위치: `/data/index/...` (docker-compose의 `app_data` 볼륨으로 영속화)

## 4) 호출 예시

```bash
# Basic RAG
curl -X POST 'http://localhost:8000/rag/basic' \
  -H 'Content-Type: application/json' \
  -d '{"query":"시드니에서 아이와 가볼만한 곳?", "thread_id":"r1"}'

# HQ-RAG
curl -X POST 'http://localhost:8000/rag/hq' \
  -H 'Content-Type: application/json' \
  -d '{"query":"시드니 가족 일정 하이라이트 추천", "thread_id":"r2"}'

# GraphRAG
curl -X POST 'http://localhost:8000/rag/graph' \
  -H 'Content-Type: application/json' \
  -d '{"query":"타롱가 동물원과 관련된 장소와 활동은?", "thread_id":"r3"}'
```

---

# 레포에 추가된 파일(핵심)

```
app/
 ├─ main.py                         # /rag/* 엔드포인트 추가
 ├─ rag/
 │   ├─ embed.py                    # OllamaEmbeddings 래퍼 (EMBED_MODEL=nom ic-embed-text)
 │   ├─ index_basic.py              # FAISS 인덱싱 스크립트
 │   ├─ index_graph.py              # GraphRAG 그래프 구축 스크립트 (NetworkX JSON)
 │   ├─ basic_rag.py                # Basic RAG 그래프 (retrieve -> generate)
 │   ├─ hq_rag.py                   # HQ-RAG 그래프 (MQE -> RRF -> LLM rerank -> generate)
 │   └─ graph_rag.py                # GraphRAG 그래프 (seed_entities -> hop_collect -> answer)
data/
 ├─ docs/                           # 샘플 문서 2개 포함
 └─ index/                          # 인덱스/그래프 저장 경로
```

---

# 각 RAG 설계 및 코드 포인트

## 1) Basic RAG (FAISS + 임베딩 + 단순 상위 k)

* **인덱싱**: `index_basic.py`

  * `RecursiveCharacterTextSplitter(chunk_size=600, overlap=120)`
  * `OllamaEmbeddings(model="nomic-embed-text")` → `FAISS.from_documents` → `save_local(...)`
* **그래프**: `basic_rag.py`

  * 노드 `retrieve`에서 `similarity_search(k=4)`로 스니펫 결합
  * 노드 `generate`에서 **컨텍스트 기반 생성**(“주어진 컨텍스트만 사용” 시스템 프롬프트)

**장점**: 간단/빠름.
**주의**: 질의 다양성/표현 차이에 취약.

## 2) HQ-RAG (멀티쿼리 + RRF + LLM 재랭킹)

* **핵심 단계**

  1. **멀티쿼리 확장(MQE)**: LLM이 `JSON 배열`로 변형 쿼리 3개 생성
  2. **RRF 융합**: 각 쿼리 상위 k(6) 결과를 `1/(rank+1)`로 누적 점수화 → 상위 6 후보
  3. **LLM 재랭킹**: 후보 `[idx, score]` JSON으로만 출력하도록 LLM에 요구 → 상위 4개 채택
  4. **생성**: 근거 스니펫 위주로 정확한 답변 생성
* **장점**: 쿼리 난해/짧을 때 강함, 컨텍스트 품질 ↑
* **비용 최적화**:

  * **라우터/소형 LLM**(예: `qwen2.5:3b-instruct`)으로 **MQE/재랭킹** 수행
  * **생성**만 `GENERATOR_MODEL`(예: `llama3:8b-instruct`)로

## 3) GraphRAG (엔티티 그래프)

* **그래프 구축**: `index_graph.py`

  * 문서 → 청크 분할 → **LLM으로 엔티티/관계 JSON 추출** → `networkx.MultiDiGraph` 에 노드/엣지 추가 → JSON 저장
* **질의 처리**: `graph_rag.py`

  1. **시드 엔티티 추출**(LLM이 질문에서 핵심 엔티티 JSON 배열로만 출력)
  2. \*\*hop 확장(기본 2-hop)\*\*으로 관련 경로 수집
  3. **경로 텍스트화**하여 **근거 중심 생성**
* **장점**: 연결/맥락 기반 추론, 관계적 질문에 강점
* **주의**: 그래프 품질은 추출 프롬프트/모델 성능에 좌우 → 배치 재구축 추천

---

# 확장 팁

* **임베딩 모델**:

  * 기본: `nomic-embed-text` (빠르고 가벼움)
  * 대안: `mxbai-embed-large`, `gte-base` 등 (정확도/속도 트레이드오프)
* **인덱스 파라미터**: 도메인/문서에 맞춰 `chunk_size/overlap` 튜닝
* **LLM 비용 절감**: HQ-RAG에서 **멀티쿼리/재랭킹**은 ROUTER\_MODEL로, **최종 생성**만 GENERATOR\_MODEL로
* **근거 표기**: 문서 경로(`d.metadata["source"]`)를 유지/노출하면 출처 트래킹 가능
* **GraphRAG 고도화**:

  * 키프레이즈 인덱스/역인덱스 혼합, **Community detection**으로 주제 subgraph 우선화
  * **질의 엔티티 disambiguation** (동명이인 분해)
  * 그래프 + 벡터 결합(하이브리드 RAG)

---

필요하시면 **웹 UI(React) 데모**, **Redis/PG 기반 영속 체크포인터**, **BM25/Hybrid 검색**, **토큰 스트리밍 결합 RAG**까지 추가 확장본도 만들어 드릴게요.

---
---
---

# 멀티턴 RAG / UI

아래는 **요약 → 멀티턴 RAG 사용법 → React 웹 UI 실행법 → 코드 포인트** 순서입니다.
요청하신 **멀티턴으로 RAG 질의응답을 여러 번 수행하는 그래프 3종(Basic/HQ/Graph)** 과 **웹 UI(React) 전체 데모 코드**를 기존 레포에 통합해서 한 번에 받을 수 있게 준비했습니다.

# 요약

* 멀티턴 RAG 그래프:

  * **Basic 멀티턴**: `retrieve → answer` (FAISS 상위 k + 컨텍스트 기반 생성, 체크포인터로 대화 맥락 계속 유지)
  * **HQ 멀티턴**: `MQE(멀티쿼리) → RRF 융합 → LLM 재랭킹 → answer` (턴마다 고품질 컨텍스트 재선정)
  * **Graph 멀티턴**: `seed(엔티티) → hop(그래프 경로 수집) → answer`
* 백엔드 FastAPI 엔드포인트(멀티턴):

  * `POST /rag/chat/basic`, `/rag/chat/hq`, `/rag/chat/graph`
* **React UI (Vite+TS+Tailwind)**: 드롭다운으로 모드 선택, `thread_id` 자동 생성, 멀티턴 대화
* 전체 프로젝트 압축본:

  * [langgraph-ollama-multiturn-demo-rag-webui.zip](sandbox:/mnt/data/langgraph-ollama-multiturn-demo-rag-webui.zip)

---

# 멀티턴 RAG — 사용법

## 0) 선행 준비

```bash
# (Docker 권장) root 디렉토리에서
cp .env.example .env
docker compose up --build

# 임베딩(권장)
docker exec -it <ollama-container-name> ollama pull nomic-embed-text
# 또는 호스트에서 ollama pull nomic-embed-text
```

## 1) 인덱싱

```bash
# Basic/HQ 공용 FAISS 인덱스
python -m app.rag.index_basic --docs /data/docs --out /data/index/faiss_basic
python -m app.rag.index_basic --docs /data/docs --out /data/index/faiss_hq

# GraphRAG 그래프 구축
python -m app.rag.index_graph  --docs /data/docs --out /data/index/graph.json
```

## 2) 멀티턴 호출 예시

동일 `thread_id`로 여러 번 호출하면 **LangGraph 체크포인터**가 대화 맥락을 유지합니다.

### Basic 멀티턴

```bash
curl -s http://localhost:8000/rag/chat/basic -H 'Content-Type: application/json' \
  -d '{"thread_id":"T1","message":"시드니에서 아이와 가볼만한 곳?"}' | jq

curl -s http://localhost:8000/rag/chat/basic -H 'Content-Type: application/json' \
  -d '{"thread_id":"T1","message":"오페라 하우스 주변 산책 코스도 추천해줘"}' | jq
```

### HQ 멀티턴

```bash
curl -s http://localhost:8000/rag/chat/hq -H 'Content-Type: application/json' \
  -d '{"thread_id":"T2","message":"시드니 가족 일정 하이라이트 알려줘"}' | jq

curl -s http://localhost:8000/rag/chat/hq -H 'Content-Type: application/json' \
  -d '{"thread_id":"T2","message":"아이 위주 활동 위주로 더 자세히"}' | jq
```

### Graph 멀티턴

```bash
curl -s http://localhost:8000/rag/chat/graph -H 'Content-Type: application/json' \
  -d '{"thread_id":"T3","message":"타롱가 동물원과 관련된 장소와 활동은?"}' | jq

curl -s http://localhost:8000/rag/chat/graph -H 'Content-Type: application/json' \
  -d '{"thread_id":"T3","message":"오페라하우스와 연결되는 경로도 보여줘"}' | jq
```

> 각 그래프는 턴마다 **재검색/재구성**하면서도, 동일 `thread_id`로 누적된 **대화 문맥**을 참고해 더 일관된 답변을 만듭니다.

---

# 웹 UI (React) — 전체 데모

## 실행

```bash
cd webui
# Node 18+ 권장
npm install
npm run dev  # http://localhost:5173
# 백엔드 FastAPI: http://localhost:8000
```

## 주요 기능

* 상단 드롭다운에서 **RAG 모드** 선택 (Basic/HQ/Graph)
* **New Chat** 클릭 시 새로운 `thread_id` 생성 → 멀티턴 세션 분리
* 메시지 입력/전송 → `POST /rag/chat/{mode}` 호출 → 응답 표시
* CORS는 FastAPI에서 활성화되어 있어 로컬 프론트엔드와 바로 연동됩니다

---

# 코드 포인트 (하이라이트)

## 백엔드 — 멀티턴 RAG 그래프

* `app/rag/chat/multiturn_basic.py`

  * `retrieve`(FAISS k=5) → `answer`(컨텍스트 기반 생성)
  * `MemorySaver` 체크포인터로 동일 `thread_id` 간 **대화 지속**
* `app/rag/chat/multiturn_hq.py`

  * `multi_query`로 질의 확장 → `RRF` 융합으로 후보 뽑기 → **LLM 재랭킹** → `answer`
* `app/rag/chat/multiturn_graph.py`

  * `seed`(질문에서 핵심 엔티티 JSON) → `hop`(2-hop 수집) → `answer`

모든 그래프는 `build()` 내부에서 `StateGraph(...).compile(checkpointer=MemorySaver())`을 사용해, **멀티턴**이 자연스럽게 이어집니다.

## 백엔드 — 엔드포인트

`app/main.py`에 아래 3개 엔드포인트 추가:

* `POST /rag/chat/basic`
* `POST /rag/chat/hq`
* `POST /rag/chat/graph`

요청 바디:

```json
{ "message": "사용자 질문", "thread_id": "세션 ID", "model": "옵션" }
```

## 프런트엔드 — React (Vite + TS + Tailwind)

* `webui/src/App.tsx` : 드롭다운/쓰레드 관리/메시지 리스트/입력란
* `webui/src/api.ts`   : 백엔드 호출 유틸
* Tailwind로 간단한 **말풍선 채팅 UI**, Enter 키 전송, 로딩 표시

---

# 확장 아이디어

1. **SSE 토큰 스트리밍**

   * 현재는 요청-응답 방식입니다. 필요 시 `/sse/updates` 또는 `/sse/values` 형식의 **LangGraph 스트리밍**을 RAG 그래프에도 붙여, 토큰/노드 업데이트를 UI에 실시간 반영 가능.

2. **Hybrid 검색**

   * FAISS(벡터) + BM25(키워드) 융합(RRF)으로 리콜/정확도 개선.

3. **출처 표기 & 하이라이팅**

   * 문서 경로/오프셋을 응답에 포함하고, 프런트에서 스니펫 하이라이트/접기 기능.

4. **서브그래프 결합**

   * 기존 **LLM 라우터/생성 이원화**와 **RAG**를 합쳐:

     * 라우터가 “RAG 필요 여부/유형(Basic/HQ/Graph)”을 결정 → 해당 서브그래프 실행

5. **체크포인터 영속화**

   * `SqliteSaver` 또는 Redis/PostgreSQL 체크포인터로 교체해 **재배포/스케일아웃** 시에도 세션 유지.

---

필요하시면 **SSE 실시간 스트리밍 UI**, **하이브리드 검색/BM25**, **문서 업로드 & 온라인 인덱싱 화면**, **Redis/PG 체크포인터**를 포함한 프로덕션 템플릿까지 확장해 드릴게요.

---
---
---

# 멀티턴 Web UI 실시간 스트리밍

아래는 **요약 → 바로 실행 가이드 → 변경 사항(백엔드/프런트) → 코드 하이라이트 → 운영 팁** 순서입니다.
요청하신 1) **SSE 실시간 스트리밍 UI**, 2) **하이브리드 검색(BM25+FAISS)**, 3) **문서 업로드 & 온라인 인덱싱**, 4) **Redis/PG 체크포인터**를 모두 기존 레포에 통합한 완성본입니다.

# 요약

* **SSE 스트리밍 UI**: `/sse/rag?mode=basic|hybrid&q=...`를 통해 **토큰 단위로 실시간** 답변 표시. 프런트에서 **Stream SSE** 체크박스로 사용.
* **하이브리드 검색**: `rank-bm25` + FAISS 결합. BM25/벡터 점수를 정규화 후 **max-pool 융합**. 엔드포인트 `/rag/hybrid`, SSE `/sse/rag?mode=hybrid`.
* **업로드 & 온라인 인덱싱**: `POST /admin/upload`로 `.txt/.md` 업로드 → `POST /admin/reindex`로 **Basic/HQ/Hybrid 인덱스 재구축**. 프런트에서 파일 선택만 하면 자동 처리.
* **Redis/PG 체크포인터**: `CHKPT_BACKEND=redis|postgres` 로 변경 가능. 샘플 그래프 `checkpointers` 제공.

**완성 ZIP:** [langgraph-ollama-multiturn-demo-rag-webui-pro.zip](sandbox:/mnt/data/langgraph-ollama-multiturn-demo-rag-webui-pro.zip)

---

# 바로 실행 가이드

## 1) Docker Compose (권장)

```bash
# 압축 해제 후
cp .env.example .env
# 필요 시 모델/임베딩 설정 수정 (EMBED_MODEL=... 등)
docker compose up --build
```

* 백엔드: [http://localhost:8000](http://localhost:8000) (`/docs`에 OpenAPI)
* Ollama: [http://localhost:11434](http://localhost:11434)

## 2) 임베딩/모델 준비(예)

```bash
# (컨테이너 또는 호스트에서)
ollama pull nomic-embed-text
ollama pull llama3:8b-instruct
ollama pull qwen2.5:3b-instruct
```

## 3) 인덱싱

```bash
# Basic/HQ
python -m app.rag.index_basic --docs /data/docs --out /data/index/faiss_basic
python -m app.rag.index_basic --docs /data/docs --out /data/index/faiss_hq

# Hybrid
python -m app.rag.hybrid.index_hybrid --docs /data/docs --faiss_out /data/index/faiss_hybrid --bm25_out /data/index/bm25.json

# GraphRAG (선행 구축)
python -m app.rag.index_graph --docs /data/docs --out /data/index/graph.json
```

## 4) React 웹 UI

```bash
cd webui
npm install
npm run dev  # http://localhost:5173
```

* 상단에서 모드 선택(Basic/HQ/Graph/Hybrid), **Stream SSE** 토글 가능
* 아래 **문서 업로드 & 온라인 인덱싱** 섹션에서 `.txt/.md` 즉시 반영

---

# 변경 사항 개요

## 백엔드(FastAPI)

* **SSE 토큰 스트리밍**

  * `GET /sse/rag?mode=basic|hybrid&q=...&thread_id=...`

    * Basic: FAISS로 컨텍스트 생성 → `ChatOllama.stream(...)`으로 **토큰 이벤트(token)** 전송
    * Hybrid: BM25+FAISS 융합 결과로 컨텍스트 생성 → 토큰 스트림
* **하이브리드 검색**

  * 인덱서: `app/rag/hybrid/index_hybrid.py` (FAISS + BM25 JSON)
  * 런타임: `app/rag/hybrid/hybrid_rag.py` (`hybrid_search()` + `generate_answer()`)
  * API: `POST /rag/hybrid`
* **문서 업로드 & 온라인 인덱싱**

  * `POST /admin/upload` (multipart) → `/data/uploads` 저장
  * `POST /admin/reindex` : `/data/docs` + `/data/uploads` 병합하여 **Basic/HQ/Hybrid** 인덱스 재생성
* **Redis/PG 체크포인터**

  * 그래프: `app/graphs/checkpointers.py`
  * 환경변수:

    * `CHKPT_BACKEND=memory|sqlite|redis|postgres`
    * `REDIS_URL=redis://localhost:6379/0`, `REDIS_NAMESPACE=lg`
    * `POSTGRES_URL=postgresql://user:pass@host:5432/langgraph`

> `requirements.txt`에 `rank-bm25, redis, psycopg2-binary, python-multipart` 추가해두었습니다.

## 프런트엔드(React)

* **Stream SSE** 토글 추가: 스트리밍 모드일 때 Basic/Hybrid은 **토큰 단위** 즉시 렌더
* **Hybrid 모드** 추가(BM25+FAISS)
* **문서 업로드 & 온라인 인덱싱** UI: 파일 선택 → 업로드 후 자동 `/admin/reindex`
* 코드 위치:

  * `webui/src/api.ts`: `sseRag()`, `uploadDoc()`, `reindexAll()` 추가
  * `webui/src/App.tsx`: 스트리밍/업로드 UI & 로직 추가

---

# 코드 하이라이트

## SSE 스트리밍(백엔드)

```py
@app.get("/sse/rag")
async def sse_rag(mode: str, q: str, thread_id: str = "rag-stream", model: Optional[str] = None):
    ...
    def gen():
        sys = SystemMessage("한국어 RAG 어시스턴트. ... 스트리밍합니다.")
        user = HumanMessage(f"[질문]\\n{q}\\n\\n[컨텍스트]\\n{ctx}")
        chat = ChatOllama(model=mdl, temperature=0.2)
        for chunk in chat.stream([sys, user]):
            part = getattr(chunk, "content", None)
            if part:
                yield {"event":"token", "data": part}
        yield {"event":"done", "data":"[DONE]"}
    return EventSourceResponse(gen())
```

## Hybrid 검색 결합

```py
def hybrid_search(query: str, k=6):
    # FAISS 유사도(거리 → 점수화) + BM25 점수 정규화
    # max-pool 융합 후 상위 k 스니펫 반환
```

## 업로드 & 온라인 인덱싱

```py
@app.post("/admin/upload")
async def upload(file: UploadFile = File(...)): ...

@app.post("/admin/reindex")
def reindex():
    # /data/docs + /data/uploads 병합 후
    # Basic/HQ FAISS, Hybrid(FAISS+BM25) 모두 재구축
```

## Redis/PG 체크포인터 선택

```py
backend = os.getenv("CHKPT_BACKEND", "memory")
if backend=="redis": from langgraph.checkpoint.redis import RedisSaver; ...
elif backend=="postgres": from langgraph.checkpoint.postgres import PostgresSaver; ...
```

## 프런트: 토큰 스트리밍

```ts
const es = sseRag(mode==='hybrid'?'hybrid':'basic', threadId, u.text)
es.addEventListener('token', (e:any)=>{ ...append token... })
es.addEventListener('done', ()=> es.close())
```

---

# 운영 팁

* **인덱스 동기화**: 업로드 후 `/admin/reindex`를 자동 호출하도록 UI에 연결해 두었습니다.
* **성능/정확도**:

  * Hybrid에서 가중치 전략(RRF, sum, max-pool) 튜닝 가능
  * BM25 토크나이저/전처리(형태소 분석기 등) 적용 시 정확도 향상
* **토큰 스트리밍**: 현재 Basic/Hybrid 지원. GraphRAG 스트리밍도 동일 패턴으로 확장 가능(그래프 컨텍스트 생성 후 stream).
* **체크포인터 운영**:

  * Redis는 in-memory 고속, TTL/Namespace로 세션 관리 필요
  * Postgres는 강한 영속성/트랜잭션 — 커넥션 풀 적용 권장
* **보안**: 업로드 MIME/type 검증, 파일 크기 제한, 인증(토큰) 등 운영 환경에 반영

---

필요하시면 **WebSocket 실시간 스트리밍**, **하이라이트/출처 표기 UI**, **BM25+Vectra 하이브리드 튜너**, **문서 삭제/부분 인덱싱**(delta update), **Redis/PG 도커 컴포즈 서비스 추가**까지 이어서 확장해 드릴게요.

---
---
---

# web socket streaming, highlight, ...

아래는 **요약 → 설치/실행 → API/UI 사용법 → 코드 포인트 → 운영 팁** 순입니다.
요청하신 1) **WebSocket 실시간 스트리밍**, 2) **하이라이트/출처 표기 UI**, 3) **BM25+Vector 하이브리드 튜너**, 4) **문서 삭제/부분 인덱싱(Delta)** 를 기존 레포에 통합해 완성본을 제공드립니다.

# 요약

* **WebSocket 스트리밍**: `/ws/rag`에서 질문을 보내면 `ctx`(컨텍스트/출처) → `token`(생성 토큰) → `done` 순으로 **실시간** 도착.
* **하이라이트/출처 UI**: 응답 하단 **Sources 패널**에 점수·출처와 함께 스니펫, **질문 키워드 하이라이트** 제공.
* **하이브리드 튜너**: `weight_vec/weight_bm25` + `fuse(max|sum|rrf)` + `rrf_k` 를 **런타임**에서 조정하는 엔드포인트 & UI 슬라이더/버튼.
* **Delta 업데이트**: 텍스트를 **즉시 인덱스 추가**(FAISS add + BM25 append), **삭제 마킹** 후 `compact`로 **부분/전체 재색인**.

**완성 ZIP:** [langgraph-ollama-multiturn-demo-rag-websocket-tuner-delta.zip](sandbox:/mnt/data/langgraph-ollama-multiturn-demo-rag-websocket-tuner-delta.zip)

---

# 설치 & 실행

```bash
# 1) 압축 해제 후 루트에서
cp .env.example .env
docker compose up --build

# 2) (권장) 모델/임베딩 준비
ollama pull nomic-embed-text
ollama pull llama3:8b-instruct
ollama pull qwen2.5:3b-instruct

# 3) 인덱싱
python -m app.rag.index_basic --docs /data/docs --out /data/index/faiss_basic
python -m app.rag.hybrid.index_hybrid --docs /data/docs --faiss_out /data/index/faiss_hybrid --bm25_out /data/index/bm25.json
python -m app.rag.index_graph  --docs /data/docs --out /data/index/graph.json

# 4) 웹 UI
cd webui && npm install && npm run dev  # http://localhost:5173
```

---

# API & UI 사용법

## 1) WebSocket 실시간 스트리밍

* 서버: `ws://localhost:8000/ws/rag?mode=basic|hybrid[&model=...]`
* 클라이언트: 연결 후 **텍스트 메시지**로 질문 전송
* 수신 이벤트(JSON):

  * `{"type":"ctx","hits":[{"text","score","source"}...]}`: 선택된 컨텍스트/출처
  * `{"type":"token","delta":"..."}`: 모델 토큰 델타
  * `{"type":"done"}`: 종료

프런트는 `Use WS` 체크를 켜면 WS를 사용하고, 끄면 기존 **SSE**로 스트리밍합니다.

## 2) 하이라이트/출처 표기 UI

* `Sources` 패널: 각 스니펫에 대해 `score`, `source` 표기
* 질문 키워드를 기준으로 스니펫에 `<mark>` 하이라이트 적용

## 3) BM25+Vector 하이브리드 튜너

* 설정 저장:
  `POST /admin/hybrid/tune` (JSON)

  ```json
  {"weight_vec":0.5,"weight_bm25":0.5,"fuse":"max","rrf_k":60}
  ```
* 조회: `GET /admin/hybrid/tune`
* 프런트: 업로드 섹션 아래 **하이브리드 튜너**에서 슬라이더/버튼으로 즉시 반영

> 병합 방식
>
> * **max**: 두 점수 중 큰 값
> * **sum**: 가중 합
> * **rrf**: Reciprocal Rank Fusion (rank 기반) + 가중치

## 4) 문서 삭제/부분 인덱싱(Delta)

* **추가(텍스트 직접)**: `POST /admin/delta/add_text`

  ```json
  {"text":"여기에 문서 내용", "source":"user_upload:note.txt"}
  ```

  * FAISS 하이브리드 인덱스에 `add_texts`로 추가
  * BM25 JSON에 append
* **삭제 표시**: `POST /admin/delta/delete`

  ```json
  {"source":"user_upload:note.txt"}
  ```

  * BM25에서는 즉시 제거, FAISS는 tombstone만 기록
* **압축(재색인)**: `POST /admin/compact`

  * tombstone(삭제 목록)을 반영하여 `/data/docs` + `/data/uploads`로 **완전 재색인**

프런트 UI:

* 업로드 섹션 안 **델타 업데이트**에서

  * 텍스트 붙여넣기 → **Add Delta**
  * `source` 식별자 입력 → **Delete**
  * **Compact**로 정리 인덱스 재구축

---

# 코드 포인트 (핵심 변경)

## 백엔드

* **WebSocket**: `app/main.py`

  ```py
  @app.websocket("/ws/rag")
  async def ws_rag(websocket: WebSocket, mode: str = "basic", model: Optional[str] = None):
      await websocket.accept()
      ...
      await websocket.send_json({"type":"ctx","hits":[...]} )
      for chunk in chat.stream([...]):
          await websocket.send_json({"type":"token","delta": part})
      await websocket.send_json({"type":"done"})
  ```
* **튜너 적용**: `app/rag/hybrid/hybrid_rag.py`

  * `/data/index/hybrid_config.json` 읽어 `max|sum|rrf` 병합 + 가중치 반영
* **Delta**:

  * `POST /admin/delta/add_text`: BM25 JSON append + FAISS `add_texts(...)`
  * `POST /admin/delta/delete`: source tombstone 기록 + BM25 즉시 제거
  * `POST /admin/compact`: tombstone 반영 **완전 재색인**

## 프런트

* **WS 연결**: `wsRag()` (origin 기반 ws/wss 자동 선택)
* **Sources 패널**: 점수·출처 표시 + **질문 키워드 하이라이트**
* **튜너 & 델타 UI**: 슬라이더/버튼/텍스트 박스 + API 연동

---

# 운영 팁

* **WS vs SSE**: 브라우저/프록시 환경에 따라 둘 중 하나가 더 안정적일 수 있습니다. 둘 다 지원하므로 상황에 맞게 토글.
* **튜너**: BM25/벡터 비중을 실서비스에서 A/B로 조정해보세요. 검색 로그 기반으로 `rrf_k`와 `k`(상위 n)도 함께 튜닝 권장.
* **Delta**: 소량 추가/삭제는 `add_text`/`delete`로 빠르게 반영하고, 주기적으로 `compact` 실행.
* **보안**: 업로드/튜너/델타/컴팩트는 운영 환경에서 반드시 인증(예: JWT·IP ACL) 적용 권장.

필요하시면 **문서 출처 클릭 시 원문 뷰어**, **하이라이트 정밀도(문장 토큰화 기반)**, **부분 컴팩션(문서 단위 재색인)**, **모델별 스트리밍 성능 프로파일링**까지 확장해 드릴게요.

---
---
---

# 출처 클릭시 원문 뷰어, 하이라이트 정밀도, 부분 컴팩션, 모델별 스트리밍 성능 프로파일링

아래는 **요약 → 실행 방법 → 기능별 사용법(UI/백엔드) → 코드/구조 포인트 → 운영 팁** 순서입니다.
요청하신 1) **문서 출처 클릭 시 원문 뷰어**, 2) **문장 토큰화 기반 하이라이트(정밀도↑)**, 3) **부분 컴팩션(문서 단위 재색인)**, 4) **모델별 스트리밍 성능 프로파일링**을 기존 레포에 모두 통합했습니다.

# 요약

* **원문 뷰어**: Sources 패널에서 **Open**을 클릭하면 모달로 원문을 열람. 파일 경로/델타 문서 모두 지원.
* **문장 단위 하이라이트**: 질의어와 매칭되는 **문장만** 필터링(`matches only`) 가능, 해당 문장들에 **<mark>** 적용.
* **부분 컴팩션**: 특정 `source`에 대해서만 **FAISS 삭제 → 재색인** 수행(문서 단위). `faiss_ids.json`로 chunk-id 매핑 관리.
* **스트리밍 프로파일링**: 여러 모델에 대해 **첫 토큰 지연(ms)**, **총 시간(ms)**, **토큰 수**, **TPS** 비교.

**완성 ZIP:**
[langgraph-ollama-multiturn-demo-rag-ws-viewer-compact-profile.zip](sandbox:/mnt/data/langgraph-ollama-multiturn-demo-rag-ws-viewer-compact-profile.zip)

---

# 바로 실행

```bash
# 1) 압축 해제 후
cp .env.example .env
docker compose up --build

# 2) (권장) 모델/임베딩 준비
ollama pull nomic-embed-text
ollama pull llama3:8b-instruct
ollama pull qwen2.5:3b-instruct

# 3) 인덱싱
python -m app.rag.index_basic --docs /data/docs --out /data/index/faiss_basic
python -m app.rag.hybrid.index_hybrid --docs /data/docs --faiss_out /data/index/faiss_hybrid --bm25_out /data/index/bm25.json
python -m app.rag.index_graph --docs /data/docs --out /data/index/graph.json

# 4) 웹 UI
cd webui && npm install && npm run dev   # http://localhost:5173
```

---

# 기능별 사용법

## 1) 문서 출처 클릭 시 원문 뷰어

* **백엔드**: `GET /source?source=<식별자>`

  * `source`가 실제 파일 경로면 디스크에서 읽어 반환.
  * 파일이 아니면 `bm25.json`에서 `(sources[], texts[])` 매칭 후 반환(델타 텍스트 포함).
* **프런트**: Sources 패널 항목의 **Open** 버튼 → 모달 표시.

  * 모달 상단 `matches only` 토글로 매칭 문장만 보이게 할 수 있음(기본 on).
  * 본문은 **문장 분리 후 질의 하이라이트** 적용.

## 2) 하이라이트 정밀도(문장 토큰화 기반)

* **프런트**: `sentenceSplit()`(간단 한국어/영문 혼합 정규식) → `highlightSentences(text, query, onlyMatches)`

  * 질의어 목록을 기반으로 **문장 단위**로 필터링 및 `<mark>` 처리.
  * 단어 수준 fallback `highlight()`도 유지.

## 3) 부분 컴팩션(문서 단위 재색인)

* **엔드포인트**: `POST /admin/compact_one`

  ```json
  { "source": "경로 또는 식별자", "text": "선택: 텍스트 직접 제공시 파일대신 사용" }
  ```

  * `faiss_ids.json`에서 해당 `source`의 기존 chunk-id를 로드 → **FAISS delete(ids)** → 최신 텍스트로 chunk 재생성 → **add\_texts(ids=...)** → 저장.
  * BM25도 동일 `source` 항목 교체(제거 후 append).
* **보조**:

  * 델타 추가(`POST /admin/delta/add_text`) 시 텍스트를 `/data/uploads/delta/`에 파일로도 저장(뷰어 호환).
  * 전체 재색인은 기존 `/admin/compact` 그대로 사용.
* **인덱서 변경점**: `app/rag/hybrid/index_hybrid.py`

  * FAISS를 `from_texts(..., ids=...)`로 구축하고, \*\*`/data/index/faiss_ids.json`\*\*에 `source → [chunk_id...]` 저장.
* **주의**: 환경에 따라 `add_texts(ids=...)` 지원이 없는 langchain 버전이 있을 수 있어, 그 경우 fallback(IDs 미저장)이 동작합니다. 가급적 최신 버전을 권장합니다.

## 4) 모델별 스트리밍 성능 프로파일링

* **백엔드**: `POST /profile/stream`

  ```json
  {
    "prompt": "프로파일링용 질문",
    "models": ["llama3:8b-instruct", "qwen2.5:3b-instruct"],
    "mode": "basic" | "hybrid" | "none",
    "k": 4
  }
  ```

  * 선택 모드에 따라 컨텍스트를 구성한 뒤, 각 모델에 대해 `chat.stream()` 기준

    * `first_token_ms` (첫 chunk까지 걸린 시간)
    * `total_ms` (전체 스트리밍 소요)
    * `tokens` (간략 토큰 근사: 공백 분리)
    * `tps` (tokens / streaming 구간 시간)
* **프런트**: 페이지 하단 **부분 컴팩션 & 프로파일링** 섹션

  * 모델 리스트(쉼표 구분) 입력 → **프로파일** 버튼 → 표로 결과 확인.

---

# 코드/구조 포인트

## 백엔드(추가/갱신)

* `app/main.py`

  * `GET /source` : 원문 뷰어
  * `POST /admin/compact_one` : 문서 단위 재색인
  * `POST /profile/stream` : 모델별 스트리밍 성능 비교
  * (이전 단계 포함) `/ws/rag`, `/sse/rag`, 업로드/델타/튜너 엔드포인트 등
* `app/rag/hybrid/index_hybrid.py`

  * **FAISS ids**와 **faiss\_ids.json** 기록(문서 단위 삭제/재색인 지원)
* 델타 추가 시:

  * `/data/uploads/delta/<source_sanitized>.txt`로 **원문 저장** + FAISS ids update

## 프런트

* `webui/src/App.tsx`

  * **Sources → Open**: 모달에 원문 표시
  * **문장 하이라이트**: `sentenceSplit()` + `highlightSentences()`
  * **부분 컴팩션**: `compactOne(source)` 버튼
  * **프로파일링**: `profileStream(prompt, models, mode)` 결과 테이블
* `webui/src/api.ts`

  * `fetchSource`, `compactOne`, `profileStream` API 유틸 추가

---

# 운영 팁

* **보안**: `/admin/*` 류 엔드포인트는 프로덕션에서 인증(토큰/JWT) 적용이 필수입니다.
* **ID 일관성**: 외부 경로 변경/이동 시 `source` 식별 의도를 유지해주세요. 델타 문서는 prefix(`user_upload:`) 사용을 권장.
* **성능 측정**: 동일 프롬프트/컨텍스트 조건에서 여러 번 평균을 내고, 입력 길이/온도 설정 등도 함께 표준화하면 비교 신뢰도가 올라갑니다.
* **하이라이트**: 더 정밀한 한국어 문장 분리(예: khaiii, kss)나 토크나이저를 붙이면 품질 향상. (의존성/용량 고려)
* **부분 컴팩션**: 대용량에서는 **샤딩(문서군 별 별도 인덱스) + 상위 레벨 fusion** 전략을 권장합니다. 샤드 단위 교체가 빠릅니다.

---

필요하시면 **문서 뷰어에서 스니펫 위치 스크롤/점프**, **RRF 가중 랭크 시각화**, **프로파일링 결과의 시계열 로그 및 CSV 다운로드**, **문서 메타데이터 편집 UI**까지 확장해 드릴게요.


