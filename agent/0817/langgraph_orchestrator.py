"""
IoT AI 챗봇 - LangGraph 마이크로워크플로우 오케스트레이터 구현
"""

from typing import Dict, List, Optional, Any, TypedDict, Annotated, Literal
from enum import Enum
import asyncio
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

import logging
from redis import Redis
from kafka import KafkaProducer, KafkaConsumer
import httpx
from sqlalchemy.ext.asyncio import AsyncSession

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Core Types and States
# =============================================================================

class WorkflowType(str, Enum):
    DEVICE_REGISTRATION = "device_registration"
    TROUBLESHOOTING = "troubleshooting"
    MANUAL_GUIDANCE = "manual_guidance"
    PURCHASE_GUIDE = "purchase_guide"

class ProcessingTier(str, Enum):
    EDGE = "edge"
    FOG = "fog" 
    CLOUD = "cloud"

class ConversationState(TypedDict):
    """전체 대화 상태를 관리하는 메인 상태 클래스"""
    # 기본 정보
    conversation_id: str
    user_id: str
    session_id: str
    timestamp: str
    
    # 사용자 입력 및 의도
    user_input: str
    intent: str
    confidence: float
    processing_tier: ProcessingTier
    
    # 대화 컨텍스트
    conversation_history: List[Dict[str, Any]]
    current_workflow: Optional[WorkflowType]
    workflow_state: str
    step_count: int
    
    # 디바이스 및 환경 정보
    device_info: Dict[str, Any]
    environment_context: Dict[str, Any]
    user_preferences: Dict[str, Any]
    
    # 처리 결과
    extracted_entities: Dict[str, Any]
    retrieved_knowledge: List[Dict[str, Any]]
    generated_response: str
    required_actions: List[Dict[str, Any]]
    
    # 상태 관리
    error_state: Optional[str]
    retry_count: int
    processing_metadata: Dict[str, Any]
    
    # 워크플로우별 전용 상태
    device_registration_state: Optional[Dict[str, Any]]
    troubleshooting_state: Optional[Dict[str, Any]]
    manual_guidance_state: Optional[Dict[str, Any]]
    purchase_guide_state: Optional[Dict[str, Any]]

# =============================================================================
# Base Orchestrator and Workflow Registry
# =============================================================================

class WorkflowRegistry:
    """워크플로우 등록 및 관리"""
    
    def __init__(self):
        self.workflows: Dict[WorkflowType, StateGraph] = {}
        self.workflow_configs: Dict[WorkflowType, Dict] = {}
        
    def register_workflow(self, workflow_type: WorkflowType, 
                         graph: StateGraph, config: Dict = None):
        """새로운 워크플로우 등록"""
        self.workflows[workflow_type] = graph
        self.workflow_configs[workflow_type] = config or {}
        logger.info(f"Registered workflow: {workflow_type}")
        
    def get_workflow(self, workflow_type: WorkflowType) -> Optional[StateGraph]:
        """워크플로우 조회"""
        return self.workflows.get(workflow_type)
        
    def list_workflows(self) -> List[WorkflowType]:
        """등록된 워크플로우 목록"""
        return list(self.workflows.keys())

class MicroworkflowOrchestrator:
    """마이크로워크플로우 오케스트레이터"""
    
    def __init__(self, 
                 redis_client: Redis,
                 kafka_producer: KafkaProducer,
                 model_router: 'AdaptiveModelRouter',
                 knowledge_engine: 'KnowledgeEngine'):
        
        self.workflow_registry = WorkflowRegistry()
        self.redis_client = redis_client
        self.kafka_producer = kafka_producer
        self.model_router = model_router
        self.knowledge_engine = knowledge_engine
        
        # 체크포인터 설정 (상태 저장용)
        self.checkpointer = MemorySaver()
        
        # 워크플로우 초기화
        self._initialize_workflows()
        
    def _initialize_workflows(self):
        """모든 워크플로우 초기화 및 등록"""
        # 각 워크플로우 인스턴스 생성 및 등록
        device_reg_workflow = DeviceRegistrationWorkflow(self)
        self.workflow_registry.register_workflow(
            WorkflowType.DEVICE_REGISTRATION,
            device_reg_workflow.build_graph()
        )
        
        troubleshooting_workflow = TroubleshootingWorkflow(self)
        self.workflow_registry.register_workflow(
            WorkflowType.TROUBLESHOOTING,
            troubleshooting_workflow.build_graph()
        )
        
        manual_guide_workflow = ManualGuidanceWorkflow(self)
        self.workflow_registry.register_workflow(
            WorkflowType.MANUAL_GUIDANCE,
            manual_guide_workflow.build_graph()
        )
        
        purchase_guide_workflow = PurchaseGuideWorkflow(self)
        self.workflow_registry.register_workflow(
            WorkflowType.PURCHASE_GUIDE,
            purchase_guide_workflow.build_graph()
        )
        
        logger.info("All workflows initialized successfully")
    
    async def process_user_request(self, 
                                 user_input: str,
                                 user_id: str,
                                 device_info: Dict,
                                 session_id: Optional[str] = None) -> Dict[str, Any]:
        """사용자 요청 처리 메인 엔트리포인트"""
        
        # 세션 ID 생성 또는 기존 사용
        if not session_id:
            session_id = str(uuid.uuid4())
            
        conversation_id = f"{user_id}_{session_id}_{int(datetime.now().timestamp())}"
        
        # 초기 상태 생성
        initial_state = ConversationState(
            conversation_id=conversation_id,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            user_input=user_input,
            intent="",
            confidence=0.0,
            processing_tier=ProcessingTier.CLOUD,
            conversation_history=[],
            current_workflow=None,
            workflow_state="start",
            step_count=0,
            device_info=device_info,
            environment_context={},
            user_preferences={},
            extracted_entities={},
            retrieved_knowledge=[],
            generated_response="",
            required_actions=[],
            error_state=None,
            retry_count=0,
            processing_metadata={},
            device_registration_state=None,
            troubleshooting_state=None,
            manual_guidance_state=None,
            purchase_guide_state=None
        )
        
        try:
            # 1. 의도 분류 및 워크플로우 선택
            workflow_type = await self._classify_intent_and_select_workflow(initial_state)
            initial_state["current_workflow"] = workflow_type
            
            # 2. 선택된 워크플로우 실행
            workflow_graph = self.workflow_registry.get_workflow(workflow_type)
            if not workflow_graph:
                raise ValueError(f"Workflow not found: {workflow_type}")
                
            # 3. 워크플로우 실행
            config = RunnableConfig(
                configurable={
                    "thread_id": session_id,
                    "checkpoint_id": conversation_id
                }
            )
            
            final_state = await workflow_graph.ainvoke(
                initial_state, 
                config=config,
                checkpointer=self.checkpointer
            )
            
            # 4. 결과 처리 및 반환
            result = await self._format_response(final_state)
            
            # 5. 이벤트 발행 (분석 및 모니터링용)
            await self._publish_workflow_event(final_state)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return await self._handle_error(initial_state, e)
    
    async def _classify_intent_and_select_workflow(self, state: ConversationState) -> WorkflowType:
        """의도 분류 및 워크플로우 선택"""
        
        user_input = state["user_input"].lower()
        
        # 간단한 키워드 기반 분류 (실제로는 LLM 기반 분류 사용)
        if any(keyword in user_input for keyword in ["등록", "연결", "설정", "셋업", "register", "connect"]):
            return WorkflowType.DEVICE_REGISTRATION
        elif any(keyword in user_input for keyword in ["문제", "에러", "안됨", "고장", "trouble", "error", "fix"]):
            return WorkflowType.TROUBLESHOOTING
        elif any(keyword in user_input for keyword in ["사용법", "매뉴얼", "어떻게", "manual", "how", "guide"]):
            return WorkflowType.MANUAL_GUIDANCE
        elif any(keyword in user_input for keyword in ["구매", "추천", "호환", "buy", "recommend", "compatible"]):
            return WorkflowType.PURCHASE_GUIDE
        else:
            # 기본적으로 매뉴얼 가이드로 분류
            return WorkflowType.MANUAL_GUIDANCE
    
    async def _format_response(self, state: ConversationState) -> Dict[str, Any]:
        """최종 응답 포맷팅"""
        return {
            "conversation_id": state["conversation_id"],
            "response": state["generated_response"],
            "actions": state["required_actions"],
            "workflow_type": state["current_workflow"],
            "workflow_state": state["workflow_state"],
            "processing_metadata": state["processing_metadata"],
            "success": state["error_state"] is None
        }
    
    async def _publish_workflow_event(self, state: ConversationState):
        """워크플로우 실행 이벤트 발행"""
        event = {
            "event_type": "workflow_completed",
            "conversation_id": state["conversation_id"],
            "workflow_type": state["current_workflow"],
            "success": state["error_state"] is None,
            "step_count": state["step_count"],
            "processing_time": state["processing_metadata"].get("total_time", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            self.kafka_producer.send("workflow_events", value=json.dumps(event))
        except Exception as e:
            logger.warning(f"Failed to publish event: {str(e)}")
    
    async def _handle_error(self, state: ConversationState, error: Exception) -> Dict[str, Any]:
        """에러 처리"""
        error_response = {
            "conversation_id": state["conversation_id"],
            "response": "죄송합니다. 요청을 처리하는 중 문제가 발생했습니다. 다시 시도해주세요.",
            "actions": [{"type": "retry", "description": "다시 시도"}],
            "workflow_type": state.get("current_workflow"),
            "error": str(error),
            "success": False
        }
        
        # 에러 로그 및 메트릭
        logger.error(f"Workflow error: {error}")
        
        return error_response

# =============================================================================
# Device Registration Workflow
# =============================================================================

class DeviceRegistrationWorkflow:
    """IoT 기기 등록 워크플로우"""
    
    def __init__(self, orchestrator: MicroworkflowOrchestrator):
        self.orchestrator = orchestrator
    
    def build_graph(self) -> StateGraph:
        """기기 등록 워크플로우 그래프 구성"""
        
        workflow = StateGraph(ConversationState)
        
        # 노드 추가
        workflow.add_node("validate_prerequisites", self.validate_prerequisites)
        workflow.add_node("scan_environment", self.scan_environment) 
        workflow.add_node("check_compatibility", self.check_compatibility)
        workflow.add_node("provide_setup_guidance", self.provide_setup_guidance)
        workflow.add_node("verify_connection", self.verify_connection)
        workflow.add_node("handle_setup_errors", self.handle_setup_errors)
        workflow.add_node("finalize_registration", self.finalize_registration)
        
        # 엣지 정의 (조건부 라우팅)
        workflow.set_entry_point("validate_prerequisites")
        
        workflow.add_conditional_edges(
            "validate_prerequisites",
            self.should_continue_prerequisites,
            {
                "continue": "scan_environment",
                "missing_info": "provide_setup_guidance",
                "error": "handle_setup_errors"
            }
        )
        
        workflow.add_edge("scan_environment", "check_compatibility")
        
        workflow.add_conditional_edges(
            "check_compatibility", 
            self.should_continue_compatibility,
            {
                "compatible": "provide_setup_guidance",
                "incompatible": "handle_setup_errors",
                "need_more_info": "provide_setup_guidance"
            }
        )
        
        workflow.add_edge("provide_setup_guidance", "verify_connection")
        
        workflow.add_conditional_edges(
            "verify_connection",
            self.should_continue_verification,
            {
                "success": "finalize_registration", 
                "failed": "handle_setup_errors",
                "retry": "provide_setup_guidance"
            }
        )
        
        workflow.add_conditional_edges(
            "handle_setup_errors",
            self.should_continue_error_handling,
            {
                "retry": "provide_setup_guidance",
                "escalate": END,
                "complete": "finalize_registration"
            }
        )
        
        workflow.add_edge("finalize_registration", END)
        
        return workflow
    
    async def validate_prerequisites(self, state: ConversationState) -> ConversationState:
        """사전 요구사항 검증"""
        logger.info("Validating prerequisites for device registration")
        
        state["step_count"] += 1
        state["workflow_state"] = "validating_prerequisites"
        
        # 기기 정보 추출
        device_info = state["device_info"]
        user_input = state["user_input"]
        
        # 필수 정보 체크리스트
        required_info = {
            "device_type": None,
            "device_model": None,  
            "network_type": None,
            "mobile_app_version": device_info.get("app_version"),
            "os_version": device_info.get("os_version")
        }
        
        # LLM을 사용한 엔티티 추출
        extracted_entities = await self._extract_device_entities(user_input)
        required_info.update(extracted_entities)
        
        # 상태 업데이트
        state["extracted_entities"] = required_info
        state["device_registration_state"] = {
            "prerequisites_checked": True,
            "missing_info": [k for k, v in required_info.items() if v is None],
            "validation_time": datetime.now().isoformat()
        }
        
        return state
    
    async def scan_environment(self, state: ConversationState) -> ConversationState:
        """네트워크 환경 스캔"""
        logger.info("Scanning network environment")
        
        state["step_count"] += 1
        state["workflow_state"] = "scanning_environment"
        
        # 환경 정보 수집 (실제로는 디바이스에서 수집한 정보 사용)
        environment_info = {
            "wifi_networks": ["HomeWiFi", "GuestNetwork"],
            "bluetooth_available": True,
            "location_permission": True,
            "network_quality": "good"
        }
        
        state["environment_context"] = environment_info
        state["device_registration_state"]["environment_scanned"] = True
        
        return state
    
    async def check_compatibility(self, state: ConversationState) -> ConversationState:
        """기기 호환성 확인"""
        logger.info("Checking device compatibility")
        
        state["step_count"] += 1
        state["workflow_state"] = "checking_compatibility"
        
        device_type = state["extracted_entities"].get("device_type")
        
        # 지식베이스에서 호환성 정보 검색
        compatibility_query = f"호환성 {device_type} 앱 버전 요구사항"
        knowledge_results = await self.orchestrator.knowledge_engine.search(
            query=compatibility_query,
            filters={"category": "compatibility"}
        )
        
        state["retrieved_knowledge"] = knowledge_results
        
        # 호환성 판정 (간소화된 로직)
        compatibility_score = 0.9  # 실제로는 복잡한 로직
        
        state["device_registration_state"]["compatibility_score"] = compatibility_score
        state["device_registration_state"]["compatibility_checked"] = True
        
        return state
    
    async def provide_setup_guidance(self, state: ConversationState) -> ConversationState:
        """설정 가이드 제공"""
        logger.info("Providing setup guidance")
        
        state["step_count"] += 1
        state["workflow_state"] = "providing_guidance"
        
        # 컨텍스트 준비
        context = {
            "device_info": state["extracted_entities"],
            "environment": state["environment_context"],
            "knowledge": state["retrieved_knowledge"]
        }
        
        # LLM을 통한 응답 생성
        guidance_prompt = self._build_guidance_prompt(context)
        response = await self.orchestrator.model_router.generate_response(
            prompt=guidance_prompt,
            context=context
        )
        
        state["generated_response"] = response["text"]
        
        # 필요한 액션 정의
        actions = [
            {
                "type": "camera_open",
                "description": "QR 코드 스캔을 위해 카메라를 열어주세요",
                "required": True
            },
            {
                "type": "wifi_settings",
                "description": "WiFi 설정 페이지로 이동",
                "required": True
            }
        ]
        
        state["required_actions"] = actions
        state["device_registration_state"]["guidance_provided"] = True
        
        return state
    
    async def verify_connection(self, state: ConversationState) -> ConversationState:
        """연결 확인"""
        logger.info("Verifying device connection")
        
        state["step_count"] += 1
        state["workflow_state"] = "verifying_connection"
        
        # 연결 상태 시뮬레이션 (실제로는 디바이스로부터 받음)
        connection_status = {
            "connected": True,
            "signal_strength": 85,
            "connection_type": "wifi",
            "verification_time": datetime.now().isoformat()
        }
        
        state["device_registration_state"]["connection_status"] = connection_status
        
        if connection_status["connected"]:
            state["generated_response"] = "기기가 성공적으로 연결되었습니다! 연결 상태를 확인 중입니다..."
        else:
            state["generated_response"] = "연결에 문제가 있습니다. 다시 시도해보겠습니다."
        
        return state
    
    async def handle_setup_errors(self, state: ConversationState) -> ConversationState:
        """설정 에러 처리"""
        logger.info("Handling setup errors")
        
        state["step_count"] += 1
        state["workflow_state"] = "handling_errors"
        
        # 에러 분석 및 해결방안 제시
        error_context = state["device_registration_state"]
        
        if not error_context.get("connection_status", {}).get("connected"):
            error_type = "connection_failed"
            solution = "네트워크 설정을 다시 확인해주세요. WiFi 비밀번호가 정확한지, 신호 강도가 충분한지 확인해보겠습니다."
        else:
            error_type = "unknown_error"
            solution = "알 수 없는 문제가 발생했습니다. 고객지원팀에 연결해드릴까요?"
        
        state["generated_response"] = solution
        state["device_registration_state"]["error_type"] = error_type
        state["device_registration_state"]["solution_provided"] = True
        
        return state
    
    async def finalize_registration(self, state: ConversationState) -> ConversationState:
        """등록 완료"""
        logger.info("Finalizing device registration")
        
        state["step_count"] += 1
        state["workflow_state"] = "completed"
        
        # 등록 완료 처리
        registration_result = {
            "device_id": f"device_{uuid.uuid4().hex[:8]}",
            "registration_time": datetime.now().isoformat(),
            "success": True
        }
        
        state["device_registration_state"]["registration_result"] = registration_result
        state["generated_response"] = f"""
        🎉 기기 등록이 완료되었습니다!
        
        기기 ID: {registration_result['device_id']}
        등록 시간: {registration_result['registration_time']}
        
        이제 모든 기능을 사용할 수 있습니다. 추가 도움이 필요하시면 언제든 말씀해주세요!
        """
        
        state["required_actions"] = [
            {
                "type": "show_success",
                "description": "등록 완료 화면 표시",
                "data": registration_result
            }
        ]
        
        return state
    
    # 조건 함수들
    def should_continue_prerequisites(self, state: ConversationState) -> str:
        missing_info = state["device_registration_state"].get("missing_info", [])
        if len(missing_info) > 2:
            return "missing_info"
        elif state.get("error_state"):
            return "error"
        else:
            return "continue"
    
    def should_continue_compatibility(self, state: ConversationState) -> str:
        score = state["device_registration_state"].get("compatibility_score", 0)
        if score > 0.8:
            return "compatible"
        elif score > 0.5:
            return "need_more_info" 
        else:
            return "incompatible"
    
    def should_continue_verification(self, state: ConversationState) -> str:
        connection = state["device_registration_state"].get("connection_status", {})
        if connection.get("connected"):
            return "success"
        elif state["retry_count"] < 3:
            return "retry"
        else:
            return "failed"
    
    def should_continue_error_handling(self, state: ConversationState) -> str:
        error_type = state["device_registration_state"].get("error_type")
        if error_type == "connection_failed" and state["retry_count"] < 2:
            return "retry"
        elif error_type == "unknown_error":
            return "escalate"
        else:
            return "complete"
    
    async def _extract_device_entities(self, user_input: str) -> Dict[str, Any]:
        """사용자 입력에서 기기 관련 엔티티 추출"""
        # 실제로는 NER 모델 또는 LLM 사용
        entities = {}
        
        if "스마트" in user_input:
            entities["device_type"] = "smart_device"
        if "카메라" in user_input:
            entities["device_type"] = "camera"
        if "전구" in user_input:
            entities["device_type"] = "smart_bulb"
            
        return entities
    
    def _build_guidance_prompt(self, context: Dict) -> str:
        """설정 가이드 프롬프트 구성"""
        return f"""
        다음 정보를 바탕으로 IoT 기기 등록을 위한 단계별 가이드를 제공해주세요:
        
        기기 정보: {context['device_info']}
        네트워크 환경: {context['environment']}
        호환성 정보: {context['knowledge']}
        
        사용자가 쉽게 따라할 수 있도록 명확하고 구체적인 단계를 제시해주세요.
        """

# =============================================================================
# Troubleshooting Workflow  
# =============================================================================

class TroubleshootingWorkflow:
    """트러블슈팅 워크플로우"""
    
    def __init__(self, orchestrator: MicroworkflowOrchestrator):
        self.orchestrator = orchestrator
    
    def build_graph(self) -> StateGraph:
        """트러블슈팅 워크플로우 그래프 구성"""
        
        workflow = StateGraph(ConversationState)
        
        # 노드 추가
        workflow.add_node("analyze_symptoms", self.analyze_symptoms)
        workflow.add_node("run_diagnostics", self.run_diagnostics)
        workflow.add_node("rank_solutions", self.rank_solutions)
        workflow.add_node("provide_step_by_step_fix", self.provide_step_by_step_fix)
        workflow.add_node("verify_resolution", self.verify_resolution)
        workflow.add_node("escalate_support", self.escalate_support)
        
        # 엣지 정의
        workflow.set_entry_point("analyze_symptoms")
        workflow.add_edge("analyze_symptoms", "run_diagnostics")
        workflow.add_edge("run_diagnostics", "rank_solutions")
        workflow.add_edge("rank_solutions", "provide_step_by_step_fix")
        
        workflow.add_conditional_edges(
            "provide_step_by_step_fix",
            self.should_verify_or_continue,
            {
                "verify": "verify_resolution",
                "continue": "provide_step_by_step_fix"
            }
        )
        
        workflow.add_conditional_edges(
            "verify_resolution",
            self.should_escalate_or_complete,
            {
                "resolved": END,
                "retry": "provide_step_by_step_fix", 
                "escalate": "escalate_support"
            }
        )
        
        workflow.add_edge("escalate_support", END)
        
        return workflow
    
    async def analyze_symptoms(self, state: ConversationState) -> ConversationState:
        """증상 분석"""
        logger.info("Analyzing troubleshooting symptoms")
        
        state["step_count"] += 1
        state["workflow_state"] = "analyzing_symptoms"
        
        # 증상 분석을 위한 LLM 호출
        symptoms_prompt = f"""
        사용자가 보고한 다음 문제를 분석해주세요:
        "{state['user_input']}"
        
        다음 형식으로 분석 결과를 제공해주세요:
        - 주요 증상
        - 가능한 원인들
        - 심각도 (1-10)
        - 관련 컴포넌트
        """
        
        analysis_result = await self.orchestrator.model_router.generate_response(
            prompt=symptoms_prompt,
            context={"user_input": state["user_input"]}
        )
        
        state["troubleshooting_state"] = {
            "symptoms_analysis": analysis_result,
            "severity": 5,  # 기본값
            "analyzed_at": datetime.now().isoformat()
        }
        
        return state
    
    async def run_diagnostics(self, state: ConversationState) -> ConversationState:
        """진단 실행"""
        logger.info("Running diagnostics")
        
        state["step_count"] += 1
        state["workflow_state"] = "running_diagnostics"
        
        # 진단 체크리스트 실행
        diagnostic_results = {
            "network_connectivity": True,
            "device_power": True,
            "app_version": "latest",
            "permissions": True,
            "device_compatibility": True
        }
        
        state["troubleshooting_state"]["diagnostic_results"] = diagnostic_results
        
        return state
    
    async def rank_solutions(self, state: ConversationState) -> ConversationState:
        """해결방안 순위 매기기"""
        logger.info("Ranking potential solutions")
        
        state["step_count"] += 1 
        state["workflow_state"] = "ranking_solutions"
        
        # 지식베이스에서 해결방안 검색
        problem_context = state["troubleshooting_state"]["symptoms_analysis"]
        solutions = await self.orchestrator.knowledge_engine.search(
            query=f"해결방안 {problem_context}",
            filters={"category": "troubleshooting"}
        )
        
        # 해결방안 순위 매기기 (성공률 기반)
        ranked_solutions = [
            {
                "id": 1,
                "title": "네트워크 재연결",
                "success_rate": 0.85,
                "difficulty": "easy",
                "estimated_time": "2-3분"
            },
            {
                "id": 2, 
                "title": "앱 재시작",
                "success_rate": 0.70,
                "difficulty": "easy", 
                "estimated_time": "1분"
            },
            {
                "id": 3,
                "title": "기기 전원 재부팅",
                "success_rate": 0.90,
                "difficulty": "medium",
                "estimated_time": "5분"
            }
        ]
        
        state["troubleshooting_state"]["ranked_solutions"] = ranked_solutions
        state["troubleshooting_state"]["current_solution_index"] = 0
        
        return state
    
    async def provide_step_by_step_fix(self, state: ConversationState) -> ConversationState:
        """단계별 해결 가이드 제공"""
        logger.info("Providing step-by-step fix")
        
        state["step_count"] += 1
        state["workflow_state"] = "providing_fix"
        
        current_index = state["troubleshooting_state"]["current_solution_index"]
        solutions = state["troubleshooting_state"]["ranked_solutions"]
        
        if current_index >= len(solutions):
            state["workflow_state"] = "escalation_needed"
            state["generated_response"] = "제시된 모든 해결방안을 시도했지만 문제가 지속됩니다. 전문 지원팀에 연결해드리겠습니다."
            return state
        
        current_solution = solutions[current_index]
        
        # 단계별 가이드 생성
        fix_prompt = f"""
        다음 해결방안에 대한 자세한 단계별 가이드를 제공해주세요:
        
        해결방안: {current_solution['title']}
        예상 소요시간: {current_solution['estimated_time']}
        난이도: {current_solution['difficulty']}
        
        사용자가 쉽게 따라할 수 있도록 구체적인 단계를 제시해주세요.
        """
        
        fix_guide = await self.orchestrator.model_router.generate_response(
            prompt=fix_prompt,
            context=current_solution
        )
        
        state["generated_response"] = fix_guide["text"]
        state["troubleshooting_state"]["current_fix_guide"] = fix_guide
        
        # 필요한 액션 정의
        actions = []
        if current_solution["title"] == "네트워크 재연결":
            actions = [
                {
                    "type": "wifi_settings",
                    "description": "WiFi 설정으로 이동",
                    "required": True
                }
            ]
        elif current_solution["title"] == "앱 재시작":
            actions = [
                {
                    "type": "app_restart",
                    "description": "앱 재시작",
                    "required": True
                }
            ]
        
        state["required_actions"] = actions
        
        return state
    
    async def verify_resolution(self, state: ConversationState) -> ConversationState:
        """해결 확인"""
        logger.info("Verifying problem resolution")
        
        state["step_count"] += 1
        state["workflow_state"] = "verifying_resolution"
        
        # 사용자 피드백 대기 (실제로는 클라이언트로부터 받음)
        # 여기서는 시뮬레이션
        resolution_status = {
            "resolved": False,  # 실제로는 사용자 피드백
            "user_satisfaction": 0,
            "additional_issues": None
        }
        
        state["troubleshooting_state"]["resolution_status"] = resolution_status
        
        if not resolution_status["resolved"]:
            # 다음 해결방안으로 이동
            current_index = state["troubleshooting_state"]["current_solution_index"]
            state["troubleshooting_state"]["current_solution_index"] = current_index + 1
            state["generated_response"] = "현재 방법으로 해결되지 않았네요. 다른 해결방안을 시도해보겠습니다."
        else:
            state["generated_response"] = "문제가 해결되어 다행입니다! 추가로 도움이 필요하시면 언제든 말씀해주세요."
        
        return state
    
    async def escalate_support(self, state: ConversationState) -> ConversationState:
        """지원팀 에스컬레이션"""
        logger.info("Escalating to human support")
        
        state["step_count"] += 1
        state["workflow_state"] = "escalated"
        
        # 에스컬레이션 티켓 생성
        ticket_info = {
            "ticket_id": f"TK-{uuid.uuid4().hex[:8].upper()}",
            "priority": "medium",
            "category": "technical_support",
            "attempted_solutions": [s["title"] for s in state["troubleshooting_state"]["ranked_solutions"]],
            "user_context": state["device_info"],
            "created_at": datetime.now().isoformat()
        }
        
        state["troubleshooting_state"]["escalation_ticket"] = ticket_info
        state["generated_response"] = f"""
        전문 지원팀에 연결해드렸습니다.
        
        🎫 지원 티켓: {ticket_info['ticket_id']}
        ⏰ 예상 응답시간: 1-2시간
        📞 긴급한 경우: 고객센터 1588-0000
        
        지원팀에서 곧 연락드릴 예정입니다.
        """
        
        state["required_actions"] = [
            {
                "type": "show_ticket_info",
                "description": "지원 티켓 정보 표시",
                "data": ticket_info
            }
        ]
        
        return state
    
    # 조건 함수들
    def should_verify_or_continue(self, state: ConversationState) -> str:
        if state["workflow_state"] == "escalation_needed":
            return "verify"
        return "verify"
    
    def should_escalate_or_complete(self, state: ConversationState) -> str:
        resolution_status = state["troubleshooting_state"].get("resolution_status", {})
        
        if resolution_status.get("resolved"):
            return "resolved"
        
        current_index = state["troubleshooting_state"]["current_solution_index"]
        max_solutions = len(state["troubleshooting_state"]["ranked_solutions"])
        
        if current_index >= max_solutions:
            return "escalate"
        else:
            return "retry"

# =============================================================================
# Manual Guidance Workflow
# =============================================================================

class ManualGuidanceWorkflow:
    """매뉴얼 가이드 워크플로우"""
    
    def __init__(self, orchestrator: MicroworkflowOrchestrator):
        self.orchestrator = orchestrator
    
    def build_graph(self) -> StateGraph:
        """매뉴얼 가이드 워크플로우 그래프 구성"""
        
        workflow = StateGraph(ConversationState)
        
        workflow.add_node("understand_query", self.understand_query)
        workflow.add_node("search_manual", self.search_manual)
        workflow.add_node("provide_guidance", self.provide_guidance)
        workflow.add_node("check_understanding", self.check_understanding)
        workflow.add_node("provide_additional_help", self.provide_additional_help)
        
        workflow.set_entry_point("understand_query")
        workflow.add_edge("understand_query", "search_manual")
        workflow.add_edge("search_manual", "provide_guidance")
        
        workflow.add_conditional_edges(
            "provide_guidance",
            self.should_check_understanding,
            {
                "check": "check_understanding",
                "complete": END
            }
        )
        
        workflow.add_conditional_edges(
            "check_understanding",
            self.should_provide_additional_help,
            {
                "additional_help": "provide_additional_help",
                "complete": END
            }
        )
        
        workflow.add_edge("provide_additional_help", END)
        
        return workflow
    
    async def understand_query(self, state: ConversationState) -> ConversationState:
        """사용자 질의 이해"""
        logger.info("Understanding user query for manual guidance")
        
        state["step_count"] += 1
        state["workflow_state"] = "understanding_query"
        
        # 질의 분석
        query_analysis = await self._analyze_user_query(state["user_input"])
        
        state["manual_guidance_state"] = {
            "query_analysis": query_analysis,
            "topic": query_analysis.get("topic", "general"),
            "complexity": query_analysis.get("complexity", "medium"),
            "analyzed_at": datetime.now().isoformat()
        }
        
        return state
    
    async def search_manual(self, state: ConversationState) -> ConversationState:
        """매뉴얼 검색"""
        logger.info("Searching manual database")
        
        state["step_count"] += 1
        state["workflow_state"] = "searching_manual"
        
        topic = state["manual_guidance_state"]["topic"]
        
        # 매뉴얼 검색
        manual_results = await self.orchestrator.knowledge_engine.search(
            query=f"매뉴얼 {topic} {state['user_input']}",
            filters={"category": "manual", "topic": topic}
        )
        
        state["retrieved_knowledge"] = manual_results
        state["manual_guidance_state"]["search_results"] = manual_results
        
        return state
    
    async def provide_guidance(self, state: ConversationState) -> ConversationState:
        """가이드 제공"""
        logger.info("Providing manual guidance")
        
        state["step_count"] += 1
        state["workflow_state"] = "providing_guidance"
        
        # 컨텍스트 준비
        context = {
            "query": state["user_input"],
            "topic": state["manual_guidance_state"]["topic"],
            "knowledge": state["retrieved_knowledge"]
        }
        
        # 가이드 생성
        guidance_prompt = f"""
        다음 사용자 질문에 대해 매뉴얼 정보를 바탕으로 친절하고 정확한 답변을 제공해주세요:
        
        질문: {context['query']}
        주제: {context['topic']}
        관련 매뉴얼 정보: {context['knowledge']}
        
        단계별로 설명하고, 필요시 주의사항도 포함해주세요.
        """
        
        guidance_response = await self.orchestrator.model_router.generate_response(
            prompt=guidance_prompt,
            context=context
        )
        
        state["generated_response"] = guidance_response["text"]
        state["manual_guidance_state"]["guidance_provided"] = True
        
        return state
    
    async def check_understanding(self, state: ConversationState) -> ConversationState:
        """이해도 확인"""
        logger.info("Checking user understanding")
        
        state["step_count"] += 1
        state["workflow_state"] = "checking_understanding"
        
        # 이해도 확인 질문 생성
        understanding_check = {
            "questions": [
                "설명드린 내용이 명확하셨나요?",
                "추가로 궁금한 부분이 있으신가요?",
                "실제로 따라해보시는데 어려움은 없으셨나요?"
            ],
            "follow_up_available": True
        }
        
        state["manual_guidance_state"]["understanding_check"] = understanding_check
        
        return state
    
    async def provide_additional_help(self, state: ConversationState) -> ConversationState:
        """추가 도움 제공"""
        logger.info("Providing additional help")
        
        state["step_count"] += 1
        state["workflow_state"] = "providing_additional_help"
        
        additional_resources = {
            "video_tutorials": ["기본 설정 영상", "고급 기능 영상"],
            "related_topics": ["관련 FAQ", "유사 문제 해결"],
            "support_options": ["실시간 채팅", "전화 상담", "원격 지원"]
        }
        
        state["manual_guidance_state"]["additional_resources"] = additional_resources
        state["generated_response"] += f"\n\n📚 추가 도움말:\n- 동영상 튜토리얼 제공\n- 관련 FAQ 확인\n- 실시간 지원 연결"
        
        return state
    
    # 조건 함수들
    def should_check_understanding(self, state: ConversationState) -> str:
        complexity = state["manual_guidance_state"]["complexity"]
        return "check" if complexity in ["medium", "high"] else "complete"
    
    def should_provide_additional_help(self, state: ConversationState) -> str:
        # 실제로는 사용자 응답을 받아서 판단
        return "additional_help"  # 시뮬레이션
    
    async def _analyze_user_query(self, user_input: str) -> Dict[str, Any]:
        """사용자 질의 분석"""
        analysis = {
            "topic": "general",
            "complexity": "medium",
            "intent": "how_to_use"
        }
        
        # 간단한 키워드 기반 분석 (실제로는 더 정교한 NLP)
        if any(word in user_input for word in ["설정", "셋업", "초기"]):
            analysis["topic"] = "setup"
        elif any(word in user_input for word in ["연결", "네트워크", "WiFi"]):
            analysis["topic"] = "connectivity"
        elif any(word in user_input for word in ["기능", "사용", "조작"]):
            analysis["topic"] = "operation"
        
        return analysis

# =============================================================================
# Purchase Guide Workflow
# =============================================================================

class PurchaseGuideWorkflow:
    """구매 가이드 워크플로우"""
    
    def __init__(self, orchestrator: MicroworkflowOrchestrator):
        self.orchestrator = orchestrator
    
    def build_graph(self) -> StateGraph:
        """구매 가이드 워크플로우 그래프 구성"""
        
        workflow = StateGraph(ConversationState)
        
        workflow.add_node("assess_needs", self.assess_needs)
        workflow.add_node("build_compatibility_matrix", self.build_compatibility_matrix)
        workflow.add_node("generate_recommendations", self.generate_recommendations)
        workflow.add_node("provide_comparison", self.provide_comparison)
        workflow.add_node("assist_purchase", self.assist_purchase)
        
        workflow.set_entry_point("assess_needs")
        workflow.add_edge("assess_needs", "build_compatibility_matrix")
        workflow.add_edge("build_compatibility_matrix", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "provide_comparison")
        workflow.add_edge("provide_comparison", "assist_purchase")
        workflow.add_edge("assist_purchase", END)
        
        return workflow
    
    async def assess_needs(self, state: ConversationState) -> ConversationState:
        """사용자 요구사항 평가"""
        logger.info("Assessing user needs for purchase guide")
        
        state["step_count"] += 1
        state["workflow_state"] = "assessing_needs"
        
        # 요구사항 추출
        needs_analysis = await self._extract_purchase_requirements(state["user_input"])
        
        state["purchase_guide_state"] = {
            "needs_analysis": needs_analysis,
            "budget_range": needs_analysis.get("budget", "medium"),
            "use_cases": needs_analysis.get("use_cases", []),
            "preferences": needs_analysis.get("preferences", {}),
            "assessed_at": datetime.now().isoformat()
        }
        
        return state
    
    async def build_compatibility_matrix(self, state: ConversationState) -> ConversationState:
        """호환성 매트릭스 구성"""
        logger.info("Building compatibility matrix")
        
        state["step_count"] += 1
        state["workflow_state"] = "building_compatibility"
        
        user_devices = state["device_info"]
        needs = state["purchase_guide_state"]["needs_analysis"]
        
        # 호환성 매트릭스 생성
        compatibility_matrix = {
            "current_ecosystem": self._analyze_ecosystem(user_devices),
            "compatible_brands": ["Samsung", "LG", "Philips"],
            "compatibility_score": {},
            "requirements": needs.get("requirements", {})
        }
        
        state["purchase_guide_state"]["compatibility_matrix"] = compatibility_matrix
        
        return state
    
    async def generate_recommendations(self, state: ConversationState) -> ConversationState:
        """추천 제품 생성"""
        logger.info("Generating product recommendations")
        
        state["step_count"] += 1
        state["workflow_state"] = "generating_recommendations"
        
        compatibility = state["purchase_guide_state"]["compatibility_matrix"]
        needs = state["purchase_guide_state"]["needs_analysis"]
        
        # 추천 제품 리스트 생성
        recommendations = [
            {
                "product_id": "smart_cam_001",
                "name": "SmartCam Pro 2024",
                "brand": "Samsung",
                "price": 150000,
                "compatibility_score": 95,
                "features": ["4K recording", "Night vision", "Smart alerts"],
                "pros": ["High quality", "Easy setup", "Good app"],
                "cons": ["Pricey", "Requires subscription for cloud"]
            },
            {
                "product_id": "smart_cam_002", 
                "name": "HomeCam Essential",
                "brand": "LG",
                "price": 89000,
                "compatibility_score": 88,
                "features": ["1080p recording", "Motion detection", "Local storage"],
                "pros": ["Affordable", "No subscription", "Reliable"],
                "cons": ["Lower resolution", "Basic features"]
            }
        ]
        
        state["purchase_guide_state"]["recommendations"] = recommendations
        
        return state
    
    async def provide_comparison(self, state: ConversationState) -> ConversationState:
        """제품 비교 제공"""
        logger.info("Providing product comparison")
        
        state["step_count"] += 1
        state["workflow_state"] = "providing_comparison"
        
        recommendations = state["purchase_guide_state"]["recommendations"]
        
        # 비교표 생성
        comparison_prompt = f"""
        다음 추천 제품들을 사용자가 쉽게 비교할 수 있도록 표로 정리해주세요:
        
        {json.dumps(recommendations, ensure_ascii=False, indent=2)}
        
        가격, 기능, 호환성, 장단점을 중심으로 비교해주세요.
        """
        
        comparison_response = await self.orchestrator.model_router.generate_response(
            prompt=comparison_prompt,
            context={"recommendations": recommendations}
        )
        
        state["generated_response"] = comparison_response["text"]
        state["purchase_guide_state"]["comparison_provided"] = True
        
        return state
    
    async def assist_purchase(self, state: ConversationState) -> ConversationState:
        """구매 지원"""
        logger.info("Assisting with purchase")
        
        state["step_count"] += 1
        state["workflow_state"] = "assisting_purchase"
        
        recommendations = state["purchase_guide_state"]["recommendations"]
        
        # 구매 지원 정보
        purchase_assistance = {
            "online_stores": ["쿠팡", "네이버쇼핑", "11번가"],
            "offline_stores": ["하이마트", "전자랜드", "롯데하이마트"],
            "discount_info": "현재 진행중인 할인 이벤트 확인",
            "installation_service": "전문 설치 서비스 이용 가능"
        }
        
        state["purchase_guide_state"]["purchase_assistance"] = purchase_assistance
        
        # 구매 링크 및 추가 정보 제공
        purchase_actions = []
        for product in recommendations:
            purchase_actions.append({
                "type": "product_link",
                "description": f"{product['name']} 구매하기",
                "data": {
                    "product_id": product["product_id"],
                    "name": product["name"],
                    "price": product["price"]
                }
            })
        
        state["required_actions"] = purchase_actions
        state["generated_response"] += f"\n\n🛒 구매 지원:\n- 온라인/오프라인 매장 안내\n- 할인 정보 확인\n- 설치 서비스 연결"
        
        return state
    
    async def _extract_purchase_requirements(self, user_input: str) -> Dict[str, Any]:
        """구매 요구사항 추출"""
        requirements = {
            "budget": "medium",
            "use_cases": ["home_security"],
            "preferences": {},
            "requirements": {}
        }
        
        # 간단한 키워드 기반 추출
        if "저렴" in user_input or "싼" in user_input:
            requirements["budget"] = "low"
        elif "고급" in user_input or "프리미엄" in user_input:
            requirements["budget"] = "high"
            
        if "보안" in user_input or "감시" in user_input:
            requirements["use_cases"].append("security")
        if "반려동물" in user_input or "펫" in user_input:
            requirements["use_cases"].append("pet_monitoring")
            
        return requirements
    
    def _analyze_ecosystem(self, device_info: Dict) -> Dict[str, Any]:
        """기존 디바이스 생태계 분석"""
        ecosystem = {
            "primary_platform": "android",  # device_info에서 추출
            "existing_smart_devices": [],
            "preferred_brands": [],
            "integration_level": "basic"
        }
        
        if device_info.get("os") == "iOS":
            ecosystem["primary_platform"] = "ios"
            ecosystem["preferred_brands"].append("Apple")
        
        return ecosystem

# =============================================================================
# Supporting Services (Mock Implementations)
# =============================================================================

class AdaptiveModelRouter:
    """적응형 모델 라우터 (Mock)"""
    
    async def generate_response(self, prompt: str, context: Dict = None) -> Dict[str, Any]:
        """응답 생성 (Mock)"""
        # 실제로는 vLLM, Ollama, 또는 외부 API 호출
        return {
            "text": f"Generated response for: {prompt[:50]}...",
            "model_used": "gemma-7b",
            "tokens": 150,
            "cost": 0.001
        }

class KnowledgeEngine:
    """지식 검색 엔진 (Mock)"""
    
    async def search(self, query: str, filters: Dict = None) -> List[Dict[str, Any]]:
        """지식 검색 (Mock)"""
        # 실제로는 FAISS + GraphRAG 검색
        return [
            {
                "id": "doc_001",
                "title": f"관련 문서 - {query}",
                "content": f"이것은 {query}에 관한 내용입니다...",
                "score": 0.85,
                "source": "manual_database"
            }
        ]

# =============================================================================
# FastAPI Integration
# =============================================================================

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: Optional[str] = None
    device_info: Dict[str, Any]

class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    actions: List[Dict[str, Any]]
    workflow_type: Optional[str]
    success: bool

app = FastAPI(title="IoT AI Chatbot Orchestrator")

# 전역 오케스트레이터 인스턴스
orchestrator = None

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작시 오케스트레이터 초기화"""
    global orchestrator
    
    # Redis, Kafka 등 외부 서비스 연결
    redis_client = Redis(host="localhost", port=6379, decode_responses=True)
    kafka_producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
    model_router = AdaptiveModelRouter()
    knowledge_engine = KnowledgeEngine()
    
    orchestrator = MicroworkflowOrchestrator(
        redis_client=redis_client,
        kafka_producer=kafka_producer, 
        model_router=model_router,
        knowledge_engine=knowledge_engine
    )
    
    logger.info("Orchestrator initialized successfully")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """메인 채팅 엔드포인트"""
    
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        result = await orchestrator.process_user_request(
            user_input=request.user_input,
            user_id=request.user_id,
            device_info=request.device_info,
            session_id=request.session_id
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "workflows": orchestrator.workflow_registry.list_workflows() if orchestrator else []
    }

@app.get("/workflows")
async def list_workflows():
    """등록된 워크플로우 목록"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        
    return {
        "workflows": orchestrator.workflow_registry.list_workflows(),
        "count": len(orchestrator.workflow_registry.workflows)
    }

if __name__ == "__main__":
    import uvicorn
    
    # 개발용 서버 실행
    uvicorn.run(
        "orchestrator:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )