from typing import Dict, Any
async def translate(state: Dict[str, Any]) -> str:
    return state.get("draft", "")
