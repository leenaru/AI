from typing import Dict, Any
from server.orchestrator.nodes.intent import detect_intent
from server.orchestrator.nodes.plan import plan_steps
from server.orchestrator.nodes.tools import execute_tools
from server.orchestrator.nodes.synthesize import synthesize
from server.orchestrator.nodes.translate import translate
from server.orchestrator.nodes.respond import respond
from server.ops.metrics import trace

async def run_conversation(payload: Dict[str, Any]):
    state: Dict[str, Any] = {"input": payload, "kb_hits": []}
    with trace("conversation"):
        intent = await detect_intent(state)
        state["intent"] = intent
        plan = await plan_steps(state)
        state["plan"] = plan
        exec_out = await execute_tools(state)
        state.update(exec_out)
        draft = await synthesize(state)
        state["draft"] = draft
        final = await translate(state)
        state["final"] = final
        return await respond(state)
