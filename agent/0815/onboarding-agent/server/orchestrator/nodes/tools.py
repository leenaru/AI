from typing import Dict, Any
from server.rag.search import hybrid_search

async def execute_tools(state: Dict[str, Any]) -> Dict[str, Any]:
    plan = state.get("plan", [])
    kb_hits = []
    actions = []
    for step in plan:
        if step["tool"] == "rag_search":
            res = hybrid_search(step["args"]["q"])
            kb_hits.extend(res.get("passages", []))
        elif step["tool"] == "request_image":
            actions.append({"type": "open_camera"})
    return {"kb_hits": kb_hits, "actions": actions}
