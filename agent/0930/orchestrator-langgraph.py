# server/orchestrator.py
from typing import Dict, Any, List, Optional, AsyncGenerator, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.tools import tool
from dataclasses import dataclass, field
import asyncio
import json
from enum import Enum

# Intent Types
class Intent(Enum):
    DEVICE_ONBOARDING = "device_onboarding"
    TROUBLESHOOTING = "troubleshooting"
    VOC_REGISTRATION = "voc_registration"
    ERROR_CODE_RESOLUTION = "error_code_resolution"
    DEVICE_MANUAL = "device_manual"
    PURCHASE_GUIDE = "purchase_guide"
    DEVICE_RECOMMENDATION = "device_recommendation"
    USER_REGISTRATION = "user_registration"
    GENERAL_INQUIRY = "general_inquiry"
    UNKNOWN = "unknown"

# Agent State
@dataclass
class AgentState:
    user_id: str
    message: str
    intent: Optional[Intent] = None
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    response: str = ""
    actions: List[Dict] = field(default_factory=list)
    retrieved_docs: List[Dict] = field(default_factory=list)
    intermediate_steps: List[str] = field(default_factory=list)
    error: Optional[str] = None
    requires_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)

class AgentOrchestrator:
    def __init__(self, model_adapter, kce, settings):
        self.model_adapter = model_adapter
        self.kce = kce
        self.settings = settings
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """LangGraph 워크플로우 구성"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("intent_detection", self.detect_intent)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("validate_input", self.validate_input)
        workflow.add_node("process_onboarding", self.process_onboarding)
        workflow.add_node("process_troubleshooting", self.process_troubleshooting)
        workflow.add_node("process_voc", self.process_voc)
        workflow.add_node("process_error_code", self.process_error_code)
        workflow.add_node("process_manual", self.process_manual)
        workflow.add_node("process_general", self.process_general)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("clarification", self.request_clarification)
        workflow.add_node("fallback", self.fallback_handler)
        
        # 시작점 설정
        workflow.set_entry_point("intent_detection")
        
        # 엣지 추가 - Intent Detection 이후 분기
        workflow.add_conditional_edges(
            "intent_detection",
            self.route_by_intent,
            {
                "retrieve": "retrieve_context",
                "clarify": "clarification",
                "fallback": "fallback"
            }
        )
        
        # Context Retrieval 이후 검증
        workflow.add_edge("retrieve_context", "validate_input")
        
        # Validation 이후 Intent별 처리
        workflow.add_conditional_edges(
            "validate_input",
            self.route_to_processor,
            {
                "onboarding": "process_onboarding",
                "troubleshooting": "process_troubleshooting",
                "voc": "process_voc",
                "error_code": "process_error_code",
                "manual": "process_manual",
                "general": "process_general",
                "clarify": "clarification"
            }
        )
        
        # 각 프로세서에서 응답 생성으로
        for processor in ["process_onboarding", "process_troubleshooting", 
                         "process_voc", "process_error_code", "process_manual", 
                         "process_general", "clarification", "fallback"]:
            workflow.add_edge(processor, "generate_response")
        
        # 응답 생성 후 종료
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    async def detect_intent(self, state: AgentState) -> AgentState:
        """사용자 의도 파악"""
        prompt = f"""
        사용자 메시지를 분석하여 의도를 파악하세요.
        
        사용자 메시지: {state.message}
        이전 대화 기록: {state.history[-3:] if state.history else 'None'}
        
        가능한 의도:
        - device_onboarding: 기기 등록/추가
        - troubleshooting: 문제 해결
        - voc_registration: VOC/불만 접수
        - error_code_resolution: 에러 코드 해결
        - device_manual: 사용 방법/매뉴얼
        - purchase_guide: 구매 가이드
        - device_recommendation: 기기 추천
        - user_registration: 사용자 등록
        - general_inquiry: 일반 문의
        
        응답 형식:
        {{
            "intent": "의도",
            "confidence": 0.0-1.0,
            "entities": {{추출된 엔티티}}
        }}
        """
        
        result = await self.model_adapter.generate(prompt)
        try:
            intent_data = json.loads(result)
            state.intent = Intent(intent_data["intent"])
            state.context["confidence"] = intent_data["confidence"]
            state.context["entities"] = intent_data.get("entities", {})
            
            if intent_data["confidence"] < 0.7:
                state.requires_clarification = True
        except:
            state.intent = Intent.UNKNOWN
            state.requires_clarification = True
        
        state.intermediate_steps.append(f"Intent detected: {state.intent}")
        return state
    
    async def retrieve_context(self, state: AgentState) -> AgentState:
        """관련 컨텍스트 검색 (RAG)"""
        # KCE를 통한 문서 검색
        docs = await self.kce.search(
            query=state.message,
            intent=state.intent.value if state.intent else None,
            top_k=5
        )
        state.retrieved_docs = docs
        state.intermediate_steps.append(f"Retrieved {len(docs)} documents")
        return state
    
    async def validate_input(self, state: AgentState) -> AgentState:
        """입력 검증 및 필수 정보 확인"""
        required_info = self._get_required_info(state.intent)
        missing_info = []
        
        for info in required_info:
            if info not in state.context.get("entities", {}):
                missing_info.append(info)
        
        if missing_info:
            state.requires_clarification = True
            state.clarification_questions = [
                f"{info}를 알려주세요." for info in missing_info
            ]
        
        state.intermediate_steps.append(f"Validation complete. Missing: {missing_info}")
        return state
    
    async def process_onboarding(self, state: AgentState) -> AgentState:
        """기기 온보딩 처리"""
        # 규칙 기반 온보딩 플로우
        device_type = state.context.get("entities", {}).get("device_type")
        
        if not device_type:
            state.response = "어떤 기기를 등록하시려고 하나요?"
            state.actions.append({
                "type": "camera_scan",
                "description": "기기의 QR코드나 모델명을 촬영해주세요"
            })
        else:
            # 온보딩 단계별 처리
            state.response = f"{device_type} 기기 등록을 시작합니다."
            state.actions.append({
                "type": "start_onboarding",
                "device": device_type,
                "steps": ["wifi_setup", "device_pairing", "registration"]
            })
        
        return state
    
    async def process_troubleshooting(self, state: AgentState) -> AgentState:
        """문제 해결 처리"""
        problem = state.context.get("entities", {}).get("problem_description")
        
        # RAG에서 가져온 문서 활용
        relevant_solutions = []
        for doc in state.retrieved_docs:
            if doc.get("type") == "troubleshooting":
                relevant_solutions.append(doc["content"])
        
        if relevant_solutions:
            prompt = f"""
            사용자 문제: {problem or state.message}
            관련 해결책: {relevant_solutions}
            
            사용자 친화적인 문제 해결 가이드를 제공하세요.
            """
            state.response = await self.model_adapter.generate(prompt)
        else:
            state.response = "문제 해결을 위해 추가 정보가 필요합니다. 어떤 문제가 발생했는지 자세히 설명해주세요."
        
        return state
    
    async def process_voc(self, state: AgentState) -> AgentState:
        """VOC 접수 처리"""
        voc_content = state.message
        
        # VOC 카테고리 분류
        category_prompt = f"""
        VOC 내용을 분석하여 카테고리를 분류하세요.
        VOC: {voc_content}
        
        카테고리: [제품불량, 서비스불만, 개선요청, 칭찬, 기타]
        우선순위: [높음, 중간, 낮음]
        """
        
        classification = await self.model_adapter.generate(category_prompt)
        
        state.response = "고객님의 소중한 의견 감사합니다. VOC가 정상적으로 접수되었습니다."
        state.actions.append({
            "type": "voc_submit",
            "content": voc_content,
            "classification": classification,
            "ticket_id": f"VOC-{state.user_id}-{len(state.history)}"
        })
        
        return state
    
    async def process_error_code(self, state: AgentState) -> AgentState:
        """에러 코드 해결"""
        error_code = state.context.get("entities", {}).get("error_code")
        
        if error_code:
            # 에러 코드 데이터베이스 조회
            solution = await self.kce.get_error_solution(error_code)
            if solution:
                state.response = f"에러 코드 {error_code}: {solution}"
            else:
                state.response = f"에러 코드 {error_code}에 대한 정보를 찾을 수 없습니다. 고객센터에 문의해주세요."
        else:
            state.response = "에러 코드를 알려주시면 해결 방법을 안내해드리겠습니다."
            state.actions.append({
                "type": "camera_scan",
                "description": "에러 코드가 표시된 화면을 촬영해주세요"
            })
        
        return state
    
    async def process_manual(self, state: AgentState) -> AgentState:
        """매뉴얼 가이드 처리"""
        query = state.message
        
        # 매뉴얼에서 관련 내용 검색
        manual_docs = [doc for doc in state.retrieved_docs if doc.get("type") == "manual"]
        
        if manual_docs:
            prompt = f"""
            사용자 질문: {query}
            매뉴얼 내용: {manual_docs}
            
            사용자 질문에 대한 명확한 답변을 제공하세요.
            """
            state.response = await self.model_adapter.generate(prompt)
        else:
            state.response = "관련 매뉴얼을 찾을 수 없습니다. 질문을 더 구체적으로 알려주세요."
        
        return state
    
    async def process_general(self, state: AgentState) -> AgentState:
        """일반 문의 처리"""
        # LLM을 활용한 일반적인 응답 생성
        prompt = f"""
        사용자 질문: {state.message}
        컨텍스트: {state.retrieved_docs[:2] if state.retrieved_docs else 'None'}
        
        IoT 기기 고객 서비스 담당자로서 친절하고 도움이 되는 답변을 제공하세요.
        """
        
        state.response = await self.model_adapter.generate(prompt)
        return state
    
    async def request_clarification(self, state: AgentState) -> AgentState:
        """명확화 요청"""
        if state.clarification_questions:
            state.response = "도움을 드리기 위해 추가 정보가 필요합니다.\n"
            state.response += "\n".join(state.clarification_questions)
        else:
            state.response = "질문을 더 구체적으로 알려주시면 정확한 답변을 드릴 수 있습니다."
        
        return state
    
    async def fallback_handler(self, state: AgentState) -> AgentState:
        """폴백 처리"""
        state.response = """
        죄송합니다. 요청을 처리할 수 없습니다.
        다음 옵션을 선택해주세요:
        1. 기기 등록/온보딩
        2. 문제 해결
        3. 사용 방법 문의
        4. 고객센터 연결
        """
        state.actions.append({
            "type": "show_menu",
            "options": ["onboarding", "troubleshooting", "manual", "customer_service"]
        })
        return state
    
    async def generate_response(self, state: AgentState) -> AgentState:
        """최종 응답 생성 및 포맷팅"""
        # 응답 품질 개선
        if state.response and not state.error:
            # 응답에 컨텍스트 추가
            state.context["process_steps"] = state.intermediate_steps
            state.context["confidence"] = state.context.get("confidence", 1.0)
        
        return state
    
    def route_by_intent(self, state: AgentState) -> str:
        """Intent 기반 라우팅"""
        if state.requires_clarification:
            return "clarify"
        elif state.intent == Intent.UNKNOWN:
            return "fallback"
        else:
            return "retrieve"
    
    def route_to_processor(self, state: AgentState) -> str:
        """프로세서 라우팅"""
        if state.requires_clarification:
            return "clarify"
        
        intent_map = {
            Intent.DEVICE_ONBOARDING: "onboarding",
            Intent.TROUBLESHOOTING: "troubleshooting",
            Intent.VOC_REGISTRATION: "voc",
            Intent.ERROR_CODE_RESOLUTION: "error_code",
            Intent.DEVICE_MANUAL: "manual",
            Intent.PURCHASE_GUIDE: "general",
            Intent.DEVICE_RECOMMENDATION: "general",
            Intent.USER_REGISTRATION: "onboarding",
            Intent.GENERAL_INQUIRY: "general"
        }
        
        return intent_map.get(state.intent, "general")
    
    def _get_required_info(self, intent: Intent) -> List[str]:
        """Intent별 필수 정보"""
        requirements = {
            Intent.DEVICE_ONBOARDING: ["device_type", "model_number"],
            Intent.TROUBLESHOOTING: ["problem_description"],
            Intent.ERROR_CODE_RESOLUTION: ["error_code"],
            Intent.VOC_REGISTRATION: [],
            Intent.DEVICE_MANUAL: ["device_type"],
            Intent.PURCHASE_GUIDE: ["budget", "requirements"],
            Intent.DEVICE_RECOMMENDATION: ["use_case", "preferences"],
            Intent.USER_REGISTRATION: ["user_info"]
        }
        return requirements.get(intent, [])
    
    async def process_message(self, user_id: str, message: str, session: Dict, 
                             context: Optional[Dict] = None, mode: str = "on_demand") -> Dict:
        """메시지 처리"""
        # State 초기화
        initial_state = AgentState(
            user_id=user_id,
            message=message,
            context=context or {},
            history=session.get("history", [])
        )
        
        # Graph 실행
        final_state = await self.graph.ainvoke(initial_state)
        
        return {
            "message": final_state.response,
            "actions": final_state.actions,
            "context": final_state.context
        }
    
    async def process_message_stream(self, user_id: str, message: str, session: Dict,
                                    context: Optional[Dict] = None, mode: str = "on_demand") -> AsyncGenerator:
        """스트리밍 메시지 처리"""
        initial_state = AgentState(
            user_id=user_id,
            message=message,
            context=context or {},
            history=session.get("history", [])
        )
        
        # Graph 스트리밍 실행
        async for event in self.graph.astream(initial_state):
            yield {
                "type": "intermediate",
                "data": event
            }
        
        # 최종 결과
        yield {
            "type": "final",
            "message": initial_state.response,
            "actions": initial_state.actions,
            "context": initial_state.context
        }
    
    def health_check(self) -> Dict:
        """헬스 체크"""
        return {
            "status": "healthy",
            "graph_nodes": len(self.graph.nodes),
            "model_status": self.model_adapter.health_check()
        }