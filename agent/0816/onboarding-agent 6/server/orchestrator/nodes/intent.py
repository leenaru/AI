from typing import Dict, Any
from pydantic import ValidationError
from server.schemas.nlu import NLU
from server.model_adapter.router import ModelRouter

router = ModelRouter.load_from_yaml("configs/model-router.yaml")

INTENT_PROMPT = (
    "당신은 고객지원 NLU입니다. 아래 한국어 입력에서\n"
    "1) intent(label) in {faq, troubleshooting, device_registration, smalltalk, other}\n"
    "2) score(0..1)\n"
    "3) slots: {error_code, product, severity, need_image}\n"
    "4) normalized_text: 검색 친화 정규화 문장(불용어/군더더기 제거)\n"
    "5) safety: {pii:[], disallowed:false}\n"
    "를 JSON으로만 출력하세요.\n\n입력:\n{TEXT}\nJSON:\n"
)

async def detect_intent(state: Dict[str, Any]) -> Dict[str, Any]:
    payload = state["input"]
    # 1) 온디바이스 NLU 결과가 오면 우선 사용
    if "nlu" in payload:
        try:
            nlu = NLU(**payload["nlu"])
            if nlu.normalized_text:
                state["input"]["text"] = nlu.normalized_text
            return nlu.model_dump()
        except ValidationError:
            pass  # 폴백

    text = payload.get("text", "")

    # 2) 서버 LLM 폴백 (비용 최소화: 부재/오류시에만)
    try:
        out = router.generate(INTENT_PROMPT.replace("{TEXT}", text))
        import json
        nlu = NLU(**json.loads(out))
        if nlu.normalized_text:
            state["input"]["text"] = nlu.normalized_text
        return nlu.model_dump()
    except Exception:
        # 3) 최후: 경량 룰 스텁
        label = "troubleshooting" if ("오류" in text or "에러" in text) else ("device_registration" if ("등록" in text or "추가" in text) else "faq")
        return {"label": label, "score": 0.55, "slots": {}, "normalized_text": text}
