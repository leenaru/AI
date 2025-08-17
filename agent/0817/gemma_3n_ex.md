# 요약

아래에 **Gemma 3 / Gemma 3n**을 실제로 써볼 수 있는 코드를 **용도별로 10가지** 정리했습니다.
로컬(Python/Transformers, vLLM 서버, Ollama-스타일 GGUF), **온디바이스(Android/iOS: LiteRT-LM & MediaPipe LLM Inference)**, **멀티모달(텍스트+이미지)**, **스트리밍/함수호출(Function Calling)**, \*\*어댑터 레이어(백엔드 교체 가능 구조)\*\*까지 포함합니다.
각 섹션 말미에 **공식 문서/레퍼런스 링크**를 달아 확인하실 수 있게 했습니다. (모델/런타임은 2025년 기준 최신: **Gemma 3 / Gemma 3n, LiteRT-LM**). ([Google AI for Developers][1], [GitHub][2])

---

## 0) 준비 사항 (요약)

* **텍스트 전용/멀티모달**: Gemma 3(오픈 가중치).
* **온디바이스 최적화**: **Gemma 3n**(MatFormer, PLE, 오디오/비전 인코더 포함).
* **런타임**: 서버(vLLM) / 로컬(PyTorch, HF Transformers) / 온디바이스(**LiteRT-LM**, MediaPipe LLM Inference). ([Google AI for Developers][1])

---

# 자세한 코드 예제

## 1) (로컬) Python + Transformers: 기본 텍스트 생성 & 스트리밍

```python
# pip install torch transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch, threading

model_id = "google/gemma-3-1b-it"   # 예: 1B Instruct (가벼운 데모용)
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
)

prompt = "다음 문장을 공손한 한국어로 요약해 주세요: Gemma 3은 멀티모달을 지원합니다."
inputs = tok(prompt, return_tensors="pt").to(model.device)

# 스트리밍 준비
streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
gen_kwargs = dict(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
    streamer=streamer,
)

thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
thread.start()
for text in streamer:
    print(text, end="", flush=True)
thread.join()
```

* 포인트: 소형(1B\~4B)부터 시작 → 이후 12B/27B로 확장. 멀티 GPU 없이도 실험 가능.
* 참고: Gemma 3 개요, 멀티모달, 128K 컨텍스트, 함수 호출 지원. ([Google AI for Developers][1])

---

## 2) (로컬) 멀티모달(텍스트+이미지) 추론

```python
# pip install transformers pillow
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

mm_id = "google/gemma-3-4b-it"  # 멀티모달 지원 변형 예시 (모델 카탈로그 확인)
processor = AutoProcessor.from_pretrained(mm_id)
model = AutoModelForVision2Seq.from_pretrained(
    mm_id, torch_dtype="auto", device_map="auto"
)

img = Image.open("sample.jpg").convert("RGB")
inputs = processor(text="이미지의 핵심을 한국어로 설명해줘.", images=img, return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(out, skip_special_tokens=True)[0])
```

* 포인트: **이미지+텍스트** 입력. (모델 카드에서 멀티모달 지원 여부 확인)
* 배경: Gemma 3는 **이미지+텍스트 입력** 및 128K 컨텍스트를 공식 지원. ([Google AI for Developers][1])

---

## 3) (서버) vLLM로 Gemma 3 서빙 + FastAPI 엔드포인트

```bash
# 설치
pip install vllm fastapi uvicorn
```

```python
# serve.py
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

llm = LLM(model="google/gemma-3-4b-it")  # 필요 시 bigger size
app = FastAPI()

class Req(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7

@app.post("/generate")
def generate(req: Req):
    sp = SamplingParams(max_tokens=req.max_new_tokens, temperature=req.temperature, top_p=0.95)
    outs = llm.generate([req.prompt], sp)
    return {"text": outs[0].outputs[0].text}
```

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```

* 포인트: 단일 GPU에서도 고성능 서빙이 가능하다는 점이 Gemma 3의 목표 중 하나. ([The Verge][3], [blog.google][4])

---

## 4) (온디바이스/크로스플랫폼) **LiteRT-LM** C++ 미니멀 예제

> LiteRT-LM은 Google의 **온디바이스 LLM 런타임**입니다. Android / Linux / macOS / Windows에서 동작하며, **Gemma 3n** 같은 모바일 퍼스트 모델 운용에 적합합니다. ([GitHub][2], [Google AI for Developers][5])

```cpp
// g++ -std=c++17 -O2 main.cc -o demo `pkg-config --cflags --libs litert_lm`
// (실제 빌드/링크 옵션은 리포 안내에 따르세요.)
#include "litert_lm/api.h"   // 가상의 헤더명: 리포 샘플에 맞춰 include
#include <iostream>

int main() {
  lt::InitOptions opt;
  opt.model_path = "gemma3n_e2b.task";   // *.task 번들 (모델+토크나이저 묶음)
  opt.device = lt::Device::kAuto;        // Auto/GPU/NPU/CPU
  lt::Model model(opt);

  lt::GenerationConfig gen;
  gen.max_new_tokens = 128;
  gen.temperature = 0.7;
  gen.stream = true;

  auto cb = [](const std::string& piece){ std::cout << piece << std::flush; };
  model.Generate("한국어로 자기소개 한 문단 작성해줘.", gen, cb);
  std::cout << std::endl;
}
```

* **.task** 번들은 내부에 여러 TFLite 서브모델(토크나이저/텍스트/비전/오디오 등)이 포함될 수 있습니다. (모바일 배포 편의 목적) ([Medium][6])
* LiteRT 자체 개요/가이드: ([Google AI for Developers][5])
* LiteRT-LM 레포: 빌드/데모 진입점 등 안내. ([GitHub][2])

---

## 5) (Android) **MediaPipe LLM Inference API** + Gemma 3n(.task) — 스트리밍, NPU 토글

> 아래 코드는 **공식 예제의 구조를 단순화한 스켈레톤**입니다. 실제 클래스/메서드명은 사용 버전에 따라 다를 수 있으므로, 레퍼런스/샘플 프로젝트를 우선 확인해 주세요. ([Hugging Face][7])

```kotlin
// build.gradle
dependencies {
  implementation("com.google.mediapipe:tasks-genai:<latest>")
}

class LlmRepo(context: Context) {
  private val opts = LlminferenceOptions.builder()
    .setModelPath("gemma3n_e2b.task")    // assets/
    .setDelegate(LlminferenceOptions.Delegate.NNAPI) // CPU/GPU/NNAPI
    .enableStreaming(true)
    .setMaxTokens(256)
    .setTemperature(0.6f)
    .build()

  private val llm = Llminference.createFromOptions(context, opts)

  fun generate(prompt: String, onToken: (String)->Unit, onDone: ()->Unit) {
    llm.generateAsync(prompt, object: StreamingCallback {
      override fun onPartialResult(token: String) { onToken(token) }
      override fun onCompleted() { onDone() }
      override fun onError(e: Exception) { /* handle */ }
    })
  }

  fun close() { llm.close() }
}
```

* **LiteRT(옛 TFLite)** + **MediaPipe LLM Inference**는 Android에서 Gemma 3n 배포를 위한 대표 경로입니다. ([Google AI for Developers][5], [Hugging Face][7])

---

## 6) (iOS) MediaPipe 기반 LLM 추론 — Swift 스켈레톤

> iOS도 MediaPipe/AI Edge 스택으로 **.task** 번들을 로드해 스트리밍 추론이 가능합니다. (패키징/네임스페이스는 예제/버전에 따라 상이) ([Hugging Face][8])

```swift
import Foundation
import AVFoundation
import MediapipeTasksGenAI

final class LLMService {
    private var llm: MPTextGenerator!

    init() throws {
        let opts = MPTextGeneratorOptions()
        opts.modelPath = Bundle.main.path(forResource: "gemma3n_e2b", ofType: "task")
        opts.maxTokens = 256
        opts.temperature = 0.7
        opts.useNeuralEngine = true   // 가속 토글(예시)

        llm = try MPTextGenerator(options: opts)
    }

    func stream(prompt: String, onToken: @escaping (String)->Void, onDone: @escaping ()->Void) {
        try? llm.generateAsync(prompt: prompt,
            partialResult: { token in onToken(token) },
            completion: { _ in onDone() })
    }
}
```

* iOS 배포 레퍼런스: 커뮤니티 모델 카드/가이드(미디어파이프 경로 안내). ([Hugging Face][8])

---

## 7) (로컬) **함수 호출(Function Calling)** 프롬프트 & 파서 예제

Gemma 3는 **함수 호출** 워크플로를 지원합니다(출력 포맷을 강제하여 툴 호출). 간단 파서 예:

```python
import json, re
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/gemma-3-4b-it"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

tools = [
  {"name": "get_weather", "description": "날씨 조회", "parameters": {"type": "object","properties": {"city":{"type":"string"}},"required":["city"]}}
]

prompt = f"""
당신은 도우미입니다. JSON으로만 답하세요.
가능한 tools: {json.dumps(tools, ensure_ascii=False)}
사용자가 날씨를 묻는다면 아래 형식으로:
{{"tool_call": {{"name":"get_weather","arguments":{{"city":"서울"}}}}}}
그 외엔 "final_answer" 필드를 사용하세요.

사용자: 내일 서울 날씨 알려줘
"""

inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256)
text = tok.decode(out[0], skip_special_tokens=True)

m = re.search(r"\{.*\}", text, re.S)  # JSON 블록 추출
obj = json.loads(m.group(0)) if m else {"final_answer": text}
print(obj)
```

* 참고: Gemma 3 개요의 **Function calling** 지원. ([Google AI for Developers][1])

---

## 8) (어댑터 레이어) 백엔드 교체 가능한 구조 (vLLM / LiteRT-LM / 외부 HTTP)

> **서비스 로직을 모델에 독립**시키는 패턴입니다. 사용자의 이전 요청과도 부합합니다.

```python
from abc import ABC, abstractmethod
import requests, subprocess, json

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...

class VllmBackend(LLMBackend):
    def __init__(self, base_url="http://localhost:8000/generate"):
        self.url = base_url
    def generate(self, prompt):
        r = requests.post(self.url, json={"prompt": prompt, "max_new_tokens":256})
        return r.json()["text"]

class LiteRTBackend(LLMBackend):
    def __init__(self, bin_path="./litert_lm_main", task="gemma3n_e2b.task"):
        self.bin = bin_path; self.task = task
    def generate(self, prompt):
        # 간단히 subprocess로 실행해 stdout 수집 (실서비스는 C-API 바인딩 권장)
        out = subprocess.run([self.bin, "--model", self.task, "--prompt", prompt],
                             capture_output=True, text=True)
        return out.stdout

class HttpBackend(LLMBackend):
    def __init__(self, url): self.url = url
    def generate(self, prompt):
        return requests.post(self.url, json={"prompt": prompt}).json()["text"]

def run(adapter: LLMBackend, user_prompt: str):
    print(adapter.generate(user_prompt))

# 사용 예
# run(VllmBackend(), "요약해줘")
# run(LiteRTBackend(), "오프라인으로 동작하나요?")
```

* 구조 장점: **교체/확장 용이**, 디바이스/서버/서드파티 백엔드 병렬 지원.

---

## 9) (로컬) 4-bit 양자화 추론(bitsandbytes)로 메모리 절감

```python
# pip install transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")

model_id = "google/gemma-3-4b-it"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
```

* Gemma 3는 **소형\~중형 사이즈**로 단일 GPU/소수 GPU 환경을 염두에 두고 최적화됨. ([blog.google][4], [The Verge][3])

---

## 10) (온디바이스) **Gemma 3n** 특징 활용 팁

* **MatFormer**로 해상도/연산량을 유연 조절, **PLE(Per-Layer Embedding)** 캐싱으로 가속 메모리 요구량 경감 → 제한된 NPU/VRAM 환경에 유리.
* **오디오 입력**(받아쓰기/명령), **비전 인코더**(간단한 장면 이해) 포함 변형 제공.
* **E2B/E4B 등급**: 총 파라미터는 더 크지만, 가속기 메모리에 상주할 핵심 트랜스포머 가중치는 상대적으로 작게 운용. (예: E2B은 약 2B 규모만 가속기에 상주)
* **2–3GB 메모리 수준**에서도 동작 가능한 타깃. (기기/설정에 따라 상이)
* 배포 경로: **LiteRT-LM** 또는 **MediaPipe LLM Inference API**(+ .task 번들). ([Google Developers Blog][9], [Google AI for Developers][10], [InfoWorld][11])

---

# 실전 체크리스트

1. **모델 선택**: 서버(대형/멀티모달 고성능) vs 온디바이스(지연·개인정보·오프라인). ([Google AI for Developers][1])
2. **런타임**: 서버(vLLM) / 로컬(PyTorch) / 온디바이스(LiteRT-LM, MediaPipe). ([GitHub][2], [Google AI for Developers][5])
3. **패키징**: Android/iOS는 **.task** 번들 채택 추세(모듈식 TFLite 구성). ([Medium][6])
4. **가속 토글**: GPU/NPU/NNAPI/ANE 옵션 테스트(발열·지연 트레이드오프). ([Google AI for Developers][5])
5. **스트리밍**: UX 체감에 중요 — vLLM 서버 SSE, LiteRT-LM/MediaPipe 콜백. ([GitHub][2])
6. **멀티모달**: 이미지(+오디오) 입력은 Gemma 3/3n 지원 변형에서만. 모델 카드 확인. ([Google AI for Developers][1])

---

## 참고 문서 (클릭하여 확인)

* Gemma 3 모델 개요(멀티모달/128K/함수호출): ([Google AI for Developers][1])
* **Gemma 3n** 모델 개요(온디바이스·MatFormer·PLE·오디오/비전): ([Google AI for Developers][10])
* **LiteRT**(구 TFLite) 개요: ([Google AI for Developers][5])
* **LiteRT-LM**(오픈소스 런타임, C++ 데모): ([GitHub][2])
* Gemma 3 릴리스/뉴스(싱글 GPU 지향 성능): ([blog.google][4], [The Verge][3])
* 커뮤니티의 Gemma 3/3n .task/MediaPipe 예시: ([Hugging Face][8])
* .task 번들 내부 구조 관련 실무 팁: ([Medium][6])

---

필요하시면 위 코드들을 **프로젝트 스캐폴딩(빌드 스크립트, 모델 로더, 토큰 스트리밍, NPU/NNAPI/ANE 토글)** 형태로 묶어 드리겠습니다. 또한 기존에 말씀하신 **어댑터 레이어** 기준으로 vLLM/LiteRT-LM/외부 LLM을 핫스왑하는 템플릿도 확장해 드릴 수 있습니다.

[1]: https://ai.google.dev/gemma/docs/core?utm_source=chatgpt.com "Gemma 3 model overview | Google AI for Developers"
[2]: https://github.com/google-ai-edge/LiteRT-LM?utm_source=chatgpt.com "google-ai-edge/LiteRT-LM"
[3]: https://www.theverge.com/ai-artificial-intelligence/627968/google-gemma-3-open-ai-model?utm_source=chatgpt.com "Google calls Gemma 3 the most powerful AI model you can run on one GPU"
[4]: https://blog.google/technology/developers/gemma-3/?utm_source=chatgpt.com "Gemma 3: Google's new open model based on Gemini 2.0"
[5]: https://ai.google.dev/edge/litert?utm_source=chatgpt.com "LiteRT overview | Google AI Edge - Gemini API"
[6]: https://farmaker47.medium.com/unfortunately-the-procedure-for-gemma-3n-is-not-yet-documented-d546bae5f9fc?utm_source=chatgpt.com "Unfortunately the procedure for Gemma 3n is not yet ..."
[7]: https://huggingface.co/litert-community/Gemma3-1B-IT?utm_source=chatgpt.com "litert-community/Gemma3-1B-IT"
[8]: https://huggingface.co/litert-community/gemma-3-270m-it?utm_source=chatgpt.com "litert-community/gemma-3-270m-it"
[9]: https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/?utm_source=chatgpt.com "Introducing Gemma 3n: The developer guide"
[10]: https://ai.google.dev/gemma/docs/gemma-3n?utm_source=chatgpt.com "Gemma 3n model overview | Google AI for Developers"
[11]: https://www.infoworld.com/article/4019759/google-releases-gemma-3n-models-for-on-device-ai.html?utm_source=chatgpt.com "Google releases Gemma 3n models for on-device AI"

---
---
---
# On-Device examples

# 요약

온디바이스에서 **Gemma 3n**을 쓰는 실전 예제를 **플랫폼별·기능별 10가지**로 정리했습니다.
**Android(MediaPipe LLM Inference)**, **iOS(MediaPipe LLM Inference)**, **크로스플랫폼 C++(LiteRT-LM)** 을 모두 다루고, **스트리밍**, **멀티모달(텍스트+이미지)**, **함수 호출(Function Calling)**, **LoRA 적용**, **.task 번들 변환 & 배포**, \*\*성능 튜닝 포인트(MatFormer·PLE·가속기 토글)\*\*까지 포함합니다. 각 섹션 끝에 **클릭 가능한 공식 문서 링크**를 달았습니다. ([Google AI for Developers][1], [GitHub][2])

---

아래부터는 **코드와 함께 자세한 설명**입니다.

---

## 1) Android — LLM Inference API로 텍스트 생성 (기본)

**Gradle 의존성**

```gradle
dependencies {
    implementation 'com.google.mediapipe:tasks-genai:0.10.24'
}
```

**초기화 & 단발 생성**

```kotlin
import com.google.mediapipe.tasks.genai.llminference.*

// 예: ADB로 /data/local/tmp/llm/model_version.task 에 푸시한 경우
val options = LlmInferenceOptions.builder()
    .setModelPath("/data/local/tmp/llm/model_version.task")
    .setMaxTokens(512)
    .setTopK(40)
    .setTemperature(0.8f)
    .setRandomSeed(101)
    .build()

val llm = LlmInference.createFromOptions(context, options)
val result = llm.generateResponse("안녕하세요. 요약 기능을 설명해 주세요.")
println(result)
```

* MediaPipe **LLM Inference**는 온디바이스로 LLM을 실행하는 공식 경로입니다. 퀵스타트·샘플·모델 변환·LoRA 가이드를 함께 제공합니다. ([Google AI for Developers][3])

---

## 2) Android — 토큰 **스트리밍** UI 적용

```kotlin
val streamingOptions = LlmInferenceOptions.builder()
    .setModelPath("/data/local/tmp/llm/model_version.task")
    .setMaxTokens(512)
    .setTopK(40)
    .setTemperature(0.7f)
    .setResultListener({ partial, done ->
        // partial: 중간 토큰 누적 표시
        appendToUi(partial)
        if (done) showDone()
    }, { error -> showError(error) })
    .build()

val llm = LlmInference.createFromOptions(context, streamingOptions)
llm.generateResponseAsync("Gemma 3n의 특징을 한 문단으로 설명해 주세요.")
```

* `generateResponseAsync()` 로 콜백 기반 스트리밍을 간단히 구현할 수 있습니다. ([Google AI for Developers][3])

---

## 3) Android — **멀티모달(텍스트+이미지)** 프롬프트 (Gemma 3n 전용)

Gemma 3n(E2B/E4B) **미디어파이프 호환 변형**으로 **텍스트+이미지** 입력을 받을 수 있습니다.

```kotlin
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.genai.llminference.*

val bmp = loadBitmapFromAssets("burger.jpg")
val mpImage: MPImage = BitmapImageBuilder(bmp).build()

val options = LlmInferenceOptions.builder()
    .setModelPath("/data/local/tmp/llm/gemma3n_e4b.task")
    .setMaxTokens(256)
    .setTopK(10)
    .setTemperature(0.4f)
    .setMaxNumImages(1) // Gemma-3n은 세션당 이미지 1장 권장
    .build()

val llm = LlmInference.createFromOptions(context, options)

val sessionOpts = LlmInferenceSession.LlmInferenceSessionOptions.builder()
    .setGraphOptions(GraphOptions.builder().setEnableVisionModality(true).build())
    .build()

LlmInferenceSession.createFromOptions(llm, sessionOpts).use { s ->
    s.addQueryChunk("이 이미지에 보이는 물체를 설명해줘.")
    s.addImage(mpImage)
    val out = s.generateResponse()
    println(out)
}
```

* Android 가이드에 **멀티모달 프롬프트** 섹션이 포함되어 있으며, Gemma 3n(E2B/E4B) 모델 카드가 연결됩니다. **AI Edge Gallery** 앱의 *Ask Image* 데모로도 바로 확인할 수 있습니다. ([Google AI for Developers][3])

---

## 4) Android — **Function Calling**(온디바이스 툴 호출)

온디바이스에서 **함수 호출 SDK(FC SDK)** 를 LLM Inference와 함께 사용하면, 모델이 **구조화된 함수 호출**(예: `get_weather(city="서울")`)을 생성하고 앱이 실행합니다.

```kotlin
// 개념 스니펫: 실제 FC SDK 인터페이스(Formatter/Parser/Constraints 등)에 맞게 적용
val declarations = listOf(
  FunctionDecl(
    name = "get_weather",
    description = "도시의 날씨를 조회합니다",
    parameters = mapOf("city" to "string")
  )
)
val formatter = ModelFormatter.forGemma3n(declarations)
val prompt = formatter.wrapUser("내일 서울 날씨 알려줘")   // 함수 스펙이 포함된 시스템+유저 텍스트 구성
val raw = llm.generateResponse(prompt)
val parsed = OutputParser.forGemma3n(declarations).parse(raw)

if (parsed.isToolCall("get_weather")) {
    val city = parsed.arg<String>("city")
    val weather = callWeatherApi(city) // 앱 내부/온디바이스 데이터
    val finalAns = formatter.wrapToolResult(weather)
    val reply = llm.generateResponse(finalAns)
    show(reply)
}
```

* FC SDK는 **포맷터·파서·제약 디코딩** 인터페이스를 제공해 함수명/파라미터만 유효하게 나오도록 강제할 수 있습니다(온디바이스 실행 지원). ([Google AI for Developers][4])

---

## 5) Android — **LoRA** 적용(어댑터 가중치)

* **훈련**: PEFT로 LoRA 학습 → `adapter_model.safetensors` 생성
* **변환**: MediaPipe 변환기로 FlatBuffer 생성(LoRA는 **GPU 백엔드 필수**)
* **로딩**: LLM Inference 옵션에 `setLoraPath()` 지정

```python
# 변환 (Python)
import mediapipe as mp
from mediapipe.tasks.python.genai import converter

config = converter.ConversionConfig(
    backend='gpu',
    # ... (기본 모델 관련 옵션)
    lora_ckpt="adapter_model.safetensors",
    lora_rank=16,
    lora_output_tflite_file="lora_adapter.tflite",
)
converter.convert_checkpoint(config)
```

```kotlin
// Android 로딩
val options = LlmInferenceOptions.builder()
    .setModelPath("/data/local/tmp/llm/base_model.task")
    .setLoraPath("/data/local/tmp/llm/lora_adapter.tflite")
    .setMaxTokens(1000)
    .setTopK(40)
    .setTemperature(0.8f)
    .build()
val llm = LlmInference.createFromOptions(context, options)
```

* 공식 가이드가 **학습 → 변환 → 로딩** 절차(및 GPU 요건)를 명시합니다. ([Google AI for Developers][3])

---

## 6) iOS — LLM Inference로 텍스트 생성 + **스트리밍**

```ruby
# Podfile
target 'MyLlmInferenceApp' do
  use_frameworks!
  pod 'MediaPipeTasksGenAI'
  pod 'MediaPipeTasksGenAIC'
end
```

```swift
import MediaPipeTasksGenai

let modelPath = Bundle.main.path(forResource: "model", ofType: "bin") // 또는 .task 등 표 참조
let opts = LlmInferenceOptions()
opts.baseOptions.modelPath = modelPath
opts.maxTokens = 512
opts.topk = 40
opts.temperature = 0.7
opts.randomSeed = 101

let llm = try LlmInference(options: opts)

// 단발
let text = try llm.generateResponse(inputText: "요약해 주세요.")

// 스트리밍
Task {
  do {
    for try await chunk in llm.generateResponseAsync(inputText: "토큰 스트리밍 테스트") {
      print(chunk)
    }
  } catch { print("error: \(error)") }
}
```

* iOS 가이드가 **의존성(Pods)**, **동기/비동기 스트리밍** 및 **모델 호환 표(.task / .bin)** 를 제공합니다. ([Google AI for Developers][5])

---

## 7) C++ — **LiteRT-LM** 로컬 실행(크로스플랫폼)

**LiteRT-LM**은 여러 LiteRT 서브모델(토크나이저/텍스트·비전 인코더/디코더 등)을 **파이프라인으로 엮어** 온디바이스 LLM을 구동하는 오픈소스 런타임입니다.

```cpp
// 개략적 예시: 실제 빌드/헤더는 레포 샘플에 맞춰 조정
#include "litert_lm/api.h"
#include <iostream>

int main() {
  lt::InitOptions opt;
  opt.model_path = "gemma3n_e2b.task";   // .task 번들
  opt.device = lt::Device::kAuto;        // CPU/GPU/NPU 등 자동 선택
  lt::Model model(opt);

  lt::GenerationConfig gen;
  gen.max_new_tokens = 128;
  gen.temperature = 0.7;
  gen.stream = true;

  model.Generate("온디바이스로 동작 중입니다.", gen,
                 [](const std::string& t){ std::cout << t << std::flush; });
  std::cout << std::endl;
}
```

* **LiteRT-LM**: “여러 LiteRT 모델을 전처리/후처리 컴포넌트와 함께 **stitch**”하는 파이프라인 프레임워크. **LiteRT** 자체는 TFLite의 후속 런타임(고성능 온디바이스). ([GitHub][2], [Google AI for Developers][6])

---

## 8) C++ — LiteRT-LM **멀티모달 파이프라인**(비전+텍스트)

Gemma 3n은 **텍스트·비전·오디오**를 처리할 수 있도록 설계되었습니다(모델/번들에 따라). LiteRT-LM에서는 **비전 인코더 + 텍스트 디코더**를 조합한 파이프라인을 구성할 수 있습니다.

```cpp
// 개념 예시(의사 코드): 실제 API는 레포의 파이프라인 빌더 예제를 참고
lt::Pipeline p;

// 1) 토크나이저
auto tok = p.AddTokenizer("tokenizer.tflite");

// 2) 비전 인코더 (이미지 → 비주얼 토큰)
auto venc = p.AddVisionEncoder("vision_encoder.tflite");

// 3) 텍스트 디코더 (텍스트/비주얼 토큰 → 생성)
auto dec = p.AddTextDecoder("text_decoder.tflite");

// 4) 데이터 흐름 연결
p.Connect(tok.Output("ids"), dec.Input("ids"));
p.Connect(venc.Output("v_tokens"), dec.Input("v_tokens"));

auto llm = lt::Model::FromPipeline(p);
llm.Generate("이미지를 요약해줘.", { .image = LoadImage("photo.jpg") }, onToken);
```

* Gemma 3n의 **MatFormer·PLE** 기반 설계와 **멀티모달 입력**은 공식 개요/모델 카드에 명시되어 있습니다. ([Google AI for Developers][1], [Hugging Face][7])

---

## 9) **.task 번들** 변환 & 배포 파이프라인

1. (개발 단계) **모델 준비**: HF의 Gemma 3n E2B/E4B **LiteRT-LM** 번들/체크포인트를 확인하거나, 변환 스크립트로 자체 모델을 **.task**로 패키징합니다. ([Hugging Face][7])
2. (변환) MediaPipe **모델 변환기**(Python)로 LLM/LoRA를 FlatBuffer/.task로 변환합니다. (가이드의 *Model conversion* 절 참고) ([Google AI for Developers][3])
3. (테스트 배포) 개발 중에는 `adb push`로 디바이스에 올린 뒤, `setModelPath("/data/local/tmp/llm/model_version.task")` 로 사용합니다. **APK에 직접 포함하기엔 용량이 큼**(런타임 다운로드 권장). ([Google AI for Developers][3])
4. (운영 배포) 앱 첫 실행 시 CDN에서 모델 다운로드 → 무결성 확인 후 앱 내 안전한 경로에 저장 → LLM Inference/LiteRT-LM에서 경로 로드.
5. (참고) **.task 번들 제작 파이프라인** 튜토리얼(서드파티)도 유용합니다. ([Medium][8])

---

## 10) 성능·메모리 **튜닝 팁** (온디바이스 현실 가이드)

* **MatFormer 스케일**: Gemma 3n은 MatFormer로 **연산/메모리 유연 축소**가 가능—기기별로 품질↔지연의 균형점 탐색. **PLE**(Per-Layer Embedding) 캐시로 임베딩 메모리 상주 부담을 경감합니다. ([Google AI for Developers][1])
* **가속기**: Android는 기기별로 CPU/GPU/NNAPI 경로가 달라집니다(LoRA는 **GPU 전제**). iOS는 **비동기 스트리밍**으로 UI 프리즈 방지. ([Google AI for Developers][3])
* **양자화**: 4-bit(가중치) + float 활성화를 흔히 사용—S24/Pixel 등 하이엔드에서 **TTFT/토큰 속도**가 실사용 가능한 수준임(벤치표는 HF 카드 참조). ([Hugging Face][7])
* **컨텍스트/토큰**: `maxTokens`, `topK`, `temperature`, `randomSeed` 를 상황별 프로파일로 분리(요약/창작/정확성 모드). ([Google AI for Developers][3])
* **멀티모달**: Android는 `EnableVisionModality(true)`, `setMaxNumImages(1)` 등 그래프 옵션을 정확히 세팅. **텍스트가 이미지보다 앞에 오도록** 프롬프트 구성 권장. ([Google AI for Developers][3])
* **샘플 앱**: **AI Edge Gallery**에서 Ask Image/Prompt Lab/AI Chat 등 **온디바이스 데모**와 벤치마크를 확인하세요. ([Google AI for Developers][3])

---

# 빠른 실전 체크리스트

1. 모델 받기: HF의 **Gemma 3n E2B/E4B LiteRT-LM** 또는 미디어파이프 호환 변형 확인. ([Hugging Face][7])
2. Android: `tasks-genai` 추가 → `.task`를 디바이스 경로로 푸시 → `LlmInference`로 실행 → 필요하면 **스트리밍/멀티모달/LoRA/FC SDK** 단계적 적용. ([Google AI for Developers][3])
3. iOS: `MediaPipeTasksGenAI` 추가 → `LlmInference`로 **동기/비동기** 실행. ([Google AI for Developers][5])
4. C++: **LiteRT-LM**으로 **.task** 구동(크로스플랫폼), 파이프라인 구성으로 **멀티모달** 조합. ([GitHub][2])
5. 튜닝: MatFormer·PLE/양자화/가속기·멀티모달 옵션으로 **지연·발열·전력** 균형 최적화. ([Google AI for Developers][1])

---

## 참고 문서 (클릭하여 확인)

* **Gemma 3n 모델 개요**(MatFormer·PLE·멀티모달): ([Google AI for Developers][1])
* **Android: LLM Inference 가이드(스트리밍·멀티모달·LoRA·갤러리 앱)**: ([Google AI for Developers][3])
* **iOS: LLM Inference 가이드(의존성·동기/비동기 스트리밍)**: ([Google AI for Developers][5])
* **LiteRT-LM GitHub(파이프라인·C++ 런타임)** / **LiteRT 개요**: ([GitHub][2], [Google AI for Developers][6])
* **Gemma 3n LiteRT-LM 모델 카드(HF, 멀티모달·벤치마크)**: ([Hugging Face][7])

---

원하시면 위 예제들을 **프로젝트 스캐폴딩**(Android/iOS/C++ 공용 코드 조직, **어댑터 레이어**로 백엔드 교체, 빌드 스크립트, 모델 다운로드/검증, 토큰 스트리밍 UI, 멀티모달 입력 파이프라인) 형태로 묶어 드리겠습니다.

[1]: https://ai.google.dev/gemma/docs/gemma-3n?utm_source=chatgpt.com "Gemma 3n model overview | Google AI for Developers - Gemini API"
[2]: https://github.com/google-ai-edge/LiteRT-LM?utm_source=chatgpt.com "google-ai-edge/LiteRT-LM"
[3]: https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android "LLM Inference guide for Android  |  Google AI Edge  |  Google AI for Developers"
[4]: https://ai.google.dev/edge/mediapipe/solutions/genai/function_calling "AI Edge Function Calling guide  |  Google AI Edge  |  Google AI for Developers"
[5]: https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/ios "LLM Inference guide for iOS  |  Google AI Edge  |  Google AI for Developers"
[6]: https://ai.google.dev/edge/litert?utm_source=chatgpt.com "LiteRT overview | Google AI Edge - Gemini API"
[7]: https://huggingface.co/google/gemma-3n-E4B-it-litert-lm "google/gemma-3n-E4B-it-litert-lm · Hugging Face"
[8]: https://farmaker47.medium.com/pipeline-to-create-task-files-for-mediapipe-llm-inference-api-939acf29494a?utm_source=chatgpt.com "Pipeline to create .task files for MediaPipe LLM Inference API"

---
---
---

# Gemma 3n on iOS

# 요약

iOS 앱에서 **Gemma 3n**으로 **온디바이스 Q/A**를 돌리는 전 과정을 **“모델 준비 → iOS 프로젝트 세팅 → Swift 코드(스트리밍 응답) → 실행/디버그”** 순서로 정리했습니다.
가장 안전한 방법은 **MediaPipe LLM Inference(iOS)** 를 쓰는 것이며, **모델은 두 가지 경로**가 있습니다.

1. **미리 변환된 .task**(LiteRT 호환, HF 프리뷰 등)를 받아서 곧바로 넣기 — *간단하지만 프리뷰 호환성 이슈가 있을 수 있음*. ([Hugging Face][1])
2. **AI Edge Torch**로 **PyTorch 체크포인트 → .tflite → .task**를 직접 변환(권장: 텍스트 Q/A 기준) — *조금 손이 가지만 iOS 지원이 명확*. ([Google AI for Developers][2], [GitHub][3])

아래 단계는 **명령어 그대로 따라 하면 되는 형태**로 적었습니다. iOS 샘플 코드와 스트리밍 구현은 **공식 iOS 가이드** 문법을 그대로 씁니다. ([Google AI for Developers][4])

---

## 0) 사전 준비(맥)

```bash
# Xcode(앱스토어) 설치 후, 기본 개발 도구
xcode-select --install

# CocoaPods 설치
sudo gem install cocoapods

# 작업 폴더
mkdir gemma3n-ios && cd gemma3n-ios
```

* iOS에서 온디바이스 LLM을 구동하는 **공식 API는 MediaPipe LLM Inference**입니다(“iOS Guide” 문서에 프로젝트 통합, 의존성, 샘플 코드 제공). ([Google AI for Developers][4])

---

## 1) iOS 프로젝트 생성 + MediaPipe LLM Inference 연동

```bash
# 새 Xcode 프로젝트(UI: SwiftUI, Lifecycle: SwiftUI App 권장)
# Xcode GUI에서 생성: Product Name = Gemma3nQA, Language = Swift

# 프로젝트 루트로 이동 후 Pod 초기화
cd Gemma3nQA
pod init
```

**Podfile** 열고 아래를 그대로 넣습니다.

```ruby
platform :ios, '15.0'
use_frameworks!

target 'Gemma3nQA' do
  pod 'MediaPipeTasksGenAI'
  pod 'MediaPipeTasksGenAIC'
end
```

설치:

```bash
pod repo update
pod install
open Gemma3nQA.xcworkspace   # 항상 .xcworkspace 를 엽니다
```

* iOS용 **의존성(pod)**, **초기화/스트리밍 코드**는 공식 가이드에 나옵니다. ([Google AI for Developers][4])

---

## 2) 모델 준비(둘 중 하나 선택)

### A안) 미리 변환된 Gemma 3n **.task**(프리뷰) 사용

* Hugging Face의 Gemma 3n **LiteRT 프리뷰** 저장소(예: *gemma-3n-E4B-it-litert-preview*)에서 `.task`(또는 번들)를 내려받아 **앱 번들에 추가**합니다.

  ```bash
  # 예시(브라우저에서 다운로드 후):
  # Xcode > 프로젝트 > "Gemma3nQA" Target > Build Phases > Copy Bundle Resources 에 .task 추가
  ```
* **주의:** 일부 프리뷰 `.task`는 플랫폼별 지원이 다르거나 웹/안드에서만 보장되는 경우가 있습니다. iOS에서도 텍스트 Q/A는 동작했지만, 패키징 차이로 로더 이슈가 보고된 적이 있습니다(특히 웹 샘플에서). 문제가 있으면 B안을 권장합니다. ([Hugging Face][1])

### B안) **AI Edge Torch**로 직접 변환(권장, iOS 명확 지원 경로)

> 목표: Hugging Face의 **`google/gemma-3n-E2B-it`**(safetensors) → **AI Edge Torch Generative → .tflite** → **MediaPipe bundler → .task**
> *변환은 현재 Linux에서 진행 권장(CPU 변환, RAM 여유 필요)*. ([Google AI for Developers][2], [GitHub][3])

1. **Linux 변환 환경**

```bash
python -m venv venv && source venv/bin/activate
pip install ai-edge-torch  # 또는 최신 nightly: ai-edge-torch-nightly
pip install mediapipe
```

* AI Edge Torch는 **PyTorch → TFLite** 변환 및 **LLM용 Generative API**를 제공합니다(현재 CPU 변환 우선). ([GitHub][3])

2. **체크포인트 받기(HF)**

```bash
# (권장) huggingface-cli 또는 브라우저로 다운로드
# 예: google/gemma-3n-E2B-it 의 safetensors 3개와 tokenizer.model 등을 받습니다.
# 폴더 예시: ./weights/gemma-3n-E2B-it/
```

* Gemma 3n 공개 체크포인트(E2B/E4B)가 HF에 있으며, 텍스트·비전 입력을 지원(오디오/비디오는 일부 프리뷰엔 미포함). Q/A는 텍스트만으로 충분합니다. ([Hugging Face][5])

3. **TFLite 변환 스켈레톤(예시)**

```python
# convert_gemma3n.py
import torch, json
import ai_edge_torch as aiet

# 1) PyTorch 로드(의사 코드: 실제 모델 로딩은 HF 가이드/모델 카드 참고)
# from transformers import AutoModelForCausalLM, AutoTokenizer
# tok = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
# mdl = AutoModelForCausalLM.from_pretrained("google/gemma-3n-E2B-it", torch_dtype="float16").eval()

# 2) 샘플 입력(텍스트 전용 Q/A면 토크나이저 ids의 샘플 형태)
sample_inputs = (torch.randint(0, 100, (1, 16)),)  # 더미 입력

edge_model = aiet.convert(
    mdl,                     # 위에서 로드한 모델
    sample_inputs,
    quantize=True            # 필요시 양자화
)
edge_model.export("gemma3n_e2b_q.tflite")
print("Exported TFLite: gemma3n_e2b_q.tflite")
```

* *구체적인 “모델 매핑/저자링”은 AI Edge Torch Generative 문서 예제에 따르세요.* 변환 → `.tflite`까지가 1단계입니다. ([GitHub][3])

4. **.task 번들링(MediaPipe bundler)**

```python
# bundle_task.py
import mediapipe as mp
from mediapipe.tasks.python.genai import bundler

config = bundler.BundleConfig(
  tflite_model="gemma3n_e2b_q.tflite",
  tokenizer_model="tokenizer.model",   # HF에서 받은 SentencePiece
  start_token="<bos>",                 # 모델 카드의 시작 토큰
  stop_tokens=["<eos>"],               # 모델 카드의 종료 토큰(들)
  output_filename="gemma3n_e2b_q.task",
  enable_bytes_to_unicode_mapping=True
)
bundler.create_bundle(config)
print("Created: gemma3n_e2b_q.task")
```

```bash
python convert_gemma3n.py
python bundle_task.py
```

* **Bundler** 사용법과 파라미터 표는 LLM Inference의 “Model conversion” 섹션에 명확히 문서화되어 있습니다. ([Google AI for Developers][2])

5. macOS로 `.task` 파일 복사 → Xcode **Copy Bundle Resources**에 추가.

> 참고: **LoRA** 어댑터까지 쓰려면(iOS는 정적 로딩) 변환 시 LoRA 옵션을 포함해 별도의 FlatBuffer를 만들고, iOS 옵션에 `loraPath`를 지정합니다(GPU 전제). 본 Q/A 예제는 **베이스 모델만** 사용합니다. ([Google AI for Developers][4])

---

## 3) iOS 코드 — **질의/응답 + 토큰 스트리밍**

아래는 **순수 Swift** 예제입니다(UIKit/SwiftUI 어디서나 동일 로직). 공식 가이드를 바탕으로 **동기/비동기** 모두 제공합니다. ([Google AI for Developers][4])

### 3-1) LLM 래퍼(Service)

```swift
// GemmaService.swift
import Foundation
import MediaPipeTasksGenai

final class GemmaService {
    private let llm: LlmInference

    init?() {
        guard let modelPath = Bundle.main.path(forResource: "gemma3n_e2b_q", ofType: "task") else {
            print("Model not found in bundle")
            return nil
        }
        let opts = LlmInferenceOptions()
        opts.baseOptions.modelPath = modelPath
        opts.maxTokens = 512        // 필요시 조정(모델 컨텍스트와 일치 권장)
        opts.topk = 40
        opts.temperature = 0.7
        opts.randomSeed = 101       // 변동성 제어(미설정 시 온도/TopK 적용 무의미 경고)
        do {
            self.llm = try LlmInference(options: opts)
        } catch {
            print("Init error: \(error)")
            return nil
        }
    }

    // 단발 요청
    func askOnce(prompt: String) throws -> String {
        return try llm.generateResponse(inputText: prompt)
    }

    // 스트리밍(AsyncSequence)
    func askStream(prompt: String) async throws -> AsyncThrowingStream<String, Error> {
        let stream = llm.generateResponseAsync(inputText: prompt)
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    for try await partial in stream {
                        continuation.yield(partial)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
```

* `generateResponse` / `generateResponseAsync` 사용법은 iOS 가이드에 그대로 있습니다. **스트리밍**은 `for try await` 패턴으로 토큰을 순차 출력합니다. ([Google AI for Developers][4])

### 3-2) 간단한 SwiftUI UI

```swift
// ContentView.swift
import SwiftUI

struct ContentView: View {
    @State private var input: String = "서울의 가을 날씨를 한 문장으로 요약해줘."
    @State private var output: String = ""
    private let service = GemmaService()

    var body: some View {
        VStack(spacing: 16) {
            Text("Gemma 3n Q/A (On-Device)")
                .font(.title2).bold()
            TextEditor(text: $input).frame(height: 120).border(.secondary)

            HStack {
                Button("단발 요청") {
                    Task {
                        if let svc = service {
                            do { output = try svc.askOnce(prompt: input) }
                            catch { output = "Error: \(error)" }
                        } else { output = "모델 초기화 실패" }
                    }
                }
                Spacer().frame(width: 12)
                Button("스트리밍 요청") {
                    Task {
                        output.removeAll()
                        if let svc = service {
                            do {
                                let stream = try await svc.askStream(prompt: input)
                                for try await chunk in stream {
                                    output += chunk
                                }
                            } catch { output = "Error: \(error)" }
                        } else { output = "모델 초기화 실패" }
                    }
                }
            }

            ScrollView { Text(output).frame(maxWidth: .infinity, alignment: .leading) }
            Spacer()
        }
        .padding()
    }
}
```

---

## 4) 빌드 & 실행

```bash
# 실제 기기 연결(권장: iPhone 14 이상)
# Xcode 상단 기기 선택 → ▶︎ Run
```

* **주의:** LLM Inference는 실행 스레드를 **블로킹**하므로, UI 프리즈를 막으려면 **백그라운드(Task/Dispatch/NSOperation)** 에서 실행합니다(스트리밍 예제처럼). ([Google AI for Developers][4])

---

## 5) 자주 겪는 문제 & 해결

* **모델을 못 찾음**: `Bundle.main.path(forResource:..., ofType:...)` 경로 확인, **Copy Bundle Resources** 포함 여부 확인. (iOS 가이드 Quickstart의 “Download a model → Add to project” 절 참고) ([Google AI for Developers][4])
* **응답이 너무 짧다/끊긴다**: `maxTokens`는 **모델의 컨텍스트 길이와 일치**하게 잡으세요(모델 카드/번들 설명 참조). 컨텍스트보다 크게 주면 실패, 작으면 조기 중지. ([Google AI for Developers][2])
* **프리뷰 .task가 로딩 실패**: 프리뷰 포맷(다중 TFL3 분할+메타데이터) 이슈 보고가 있었습니다. **AI Edge Torch 변환 후 bundler로 만든 .task**를 권장합니다(텍스트 Q/A 기준). ([Hugging Face][1], [Google AI for Developers][2])
* **성능**: iOS는 현재 **CPU 경로 우선**(AI Edge Torch 변환본). 토큰 스트리밍으로 UX를 부드럽게 유지하세요. 필요 시 `temperature/topK/seed`를 조절합니다. ([GitHub][3], [Google AI for Developers][4])
* **LoRA 쓰고 싶다면**: iOS는 **정적 로딩**만 지원(`loraPath` 지정). 변환 시 **GPU 백엔드 전제**인 점과 지원 레이어 제한(Attention) 주의(텍스트 Q/A는 베이스 모델 권장). ([Google AI for Developers][4])

---

## 6) 공식 예제/문서(클릭해 열기)

* **iOS용 LLM Inference 가이드(Quickstart·Pod·코드·스트리밍·LoRA 섹션)**: ([Google AI for Developers][4])
* **LLM Inference(개요/모델/모델 변환·번들링 가이드)**: ([Google AI for Developers][2])
* **AI Edge Torch(파이토치→TFLite 변환 & Generative API)**: ([GitHub][3])
* **Gemma 3n 모델/체크포인트(HF, E2B/E4B)**: ([Hugging Face][5])
* **iOS 샘플 프로젝트(공식 레포)**: ([GitHub][6])

---

### 마무리 팁

* 처음엔 \*\*B안(직접 변환)\*\*으로 텍스트 Q/A 파이프라인을 확실히 성공시키고, 여유가 생기면 **A안(프리뷰 .task)** 를 시도해보세요.
* **컨텍스트 크기**(KV 캐시/맥스 토큰)와 **양자화** 설정이 실사용 성능을 좌우합니다. 모델 카드/번들 설명의 권장값을 지키세요. ([Google AI for Developers][2])

필요하시면 위 코드를 **프로젝트 스캐폴딩**(모델 다운로더, 무결성 체크, 실패 대비 폴백, 토큰 스트리밍 뷰) 형태로 정리해 드릴게요.

[1]: https://huggingface.co/google/gemma-3n-E4B-it-litert-preview/discussions/4 "google/gemma-3n-E4B-it-litert-preview · Gemma 3n does not seem to work in the sample application for web"
[2]: https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/index "LLM Inference guide  |  Google AI Edge  |  Google AI for Developers"
[3]: https://github.com/google-ai-edge/ai-edge-torch "GitHub - google-ai-edge/ai-edge-torch: Supporting PyTorch models with the Google AI Edge TFLite runtime."
[4]: https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/ios "LLM Inference guide for iOS  |  Google AI Edge  |  Google AI for Developers"
[5]: https://huggingface.co/google/gemma-3n-E2B-it/tree/main?utm_source=chatgpt.com "google/gemma-3n-E2B-it at main"
[6]: https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/llm_inference/ios "mediapipe-samples/examples/llm_inference/ios at main · google-ai-edge/mediapipe-samples · GitHub"

