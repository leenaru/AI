from typing import Dict, Any, List

async def plan_steps(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    intent = state.get("intent", {})
    label = intent.get("label")
    plan = []
    if label == "troubleshooting":
        plan.append({"tool": "rag_search", "args": {"q": state["input"].get("text", "")}})
        if any(k in state["input"].get("text", "") for k in ["사진", "이미지", "첨부"]):
            plan.append({"tool": "request_image"})
    else:
        plan.append({"tool": "rag_search", "args": {"q": state["input"].get("text", "")}})
    return plan
