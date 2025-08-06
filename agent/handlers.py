# handlers.py
from app.ollama_utils import ask_llm
from app.rag import query_rag
from app.scenario_loader_utils import TranslationManager

def detect_intent(state):
    print("[detect_intent]", state)
    user_input = state.get("user_input", "")
    lang = state.get("lang", "ko")
    t = TranslationManager(lang)

    known_intents = {
        "install_appliance": t.get("intent.install_appliance", "가전제품 설치"),
        "register_appliance": t.get("intent.register_appliance", "제품 등록"),
        "user_manual": t.get("intent.user_manual", "사용 설명서 요청"),
        "report_fault": t.get("intent.report_fault", "고장 문의 또는 접수")
    }

    prompt = t.format("prompt.detect_intent", 
        user_input=user_input, 
        options=', '.join(known_intents.keys())
    )

    intent = ask_llm(prompt).strip().lower()
    if intent not in known_intents:
        intent = "fallback"
    state["intent"] = intent
    return state

def check_info(state):
    print("[check_info]", state)
    required_fields = ["제품명", "모델명", "설치주소"]
    info = state.get("collected_info", {})
    missing = [f for f in required_fields if f not in info]
    state["info_status"] = "complete" if not missing else "incomplete"
    state["missing_fields"] = missing
    return state

def install_appliance(state):
    print("[install_appliance]", state)
    info = state.get("collected_info", {})
    if info:
        state["result"] = "installation_scheduled"
    else:
        state["result"] = "installation_failed"
    return state

def register_appliance(state):
    print("[register_appliance]", state)
    info = state.get("collected_info", {})
    if info:
        state["result"] = "registration_complete"
    else:
        state["result"] = "registration_failed"
    return state

def provide_user_manual(state):
    print("[provide_user_manual]", state)
    query = state.get("user_input", "")
    rag_response = query_rag(query)
    state["manual_link"] = rag_response
    return state

def report_fault(state):
    print("[report_fault]", state)
    query = state.get("user_input", "")
    rag_response = query_rag(query)
    state["rag_response"] = rag_response
    if "해결" in rag_response:
        state["rag_result"] = "solution_found"
    else:
        state["rag_result"] = "no_solution"
    return state

def fallback(state):
    print("[fallback]", state)
    state["error"] = "죄송합니다. 요청을 이해하지 못했습니다."
    return state

function_map = {
    "detect_intent": detect_intent,
    "check_info": check_info,
    "install_appliance": install_appliance,
    "register_appliance": register_appliance,
    "provide_user_manual": provide_user_manual,
    "report_fault": report_fault,
    "fallback": fallback,
}
