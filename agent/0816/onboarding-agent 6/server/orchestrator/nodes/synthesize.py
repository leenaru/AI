from typing import Dict, Any
from server.model_adapter.router import ModelRouter
router = ModelRouter.load_from_yaml("configs/model-router.yaml")
async def synthesize(state: Dict[str, Any]) -> str:
    prompt = f"사용자 질문: {state['input'].get('text','')}\n근거:{[p['text'] for p in state.get('kb_hits',[])][:3]}\n정확하고 간결하게 답변하라."
    return router.generate(prompt)
