from typing import Dict, Any

async def detect_intent(state: Dict[str, Any]):
    text = state["input"].get("text", "")
    score = 0.8 if any(k in text.lower() for k in ["등록", "추가", "사진", "에러", "오류"]) else 0.5
    label = "troubleshooting" if "오류" in text or "에러" in text else "faq"
    slots = {}
    return {"label": label, "score": score, "slots": slots}
