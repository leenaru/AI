from typing import Dict, Any
async def respond(state: Dict[str, Any]):
    return {"messages":[state.get("final","")],"actions":state.get("actions",[]),"citations":[{"id":h.get("id"),"meta":h.get("meta")} for h in state.get("kb_hits",[])]}
