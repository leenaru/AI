import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, asdict
from datetime import datetime
import operator

# LangGraph 및 LangChain 관련 imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Atlassian API imports
from atlassian import Confluence, Jira
import requests
from bs4 import BeautifulSoup

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 상태 타입 정의
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    confluence_content: Optional[str]
    page_info: Optional[Dict[str, Any]]
    raw_requirements: Optional[List[Dict[str, Any]]]
    analyzed_requirements: Optional[List[Dict[str, Any]]]
    user_clarifications: Optional[Dict[str, str]]
    final_requirements: Optional[List[Dict[str, Any]]]
    jira_tickets: Optional[List[Dict[str, Any]]]
    current_step: str
    user_confirmed: bool
    needs_clarification: bool
    clarification_questions: Optional[List[str]]

@dataclass
class RequirementItem:
    """요구사항 아이템 데이터 클래스"""
    title: str
    description: str
    priority: str = "Medium"
    story_points: Optional[int] = None
    acceptance_criteria: List[str] = None
    category: str = "Story"
    labels: List[str] = None
    components: List[str] = None
    confidence_score: float = 0.0
    missing_info: List[str] = None
    
    def __post_init__(self):
        if self.acceptance_criteria is None:
            self.acceptance_criteria = []
        if self.labels is None:
            self.labels = []
        if self.components is None:
            self.components = []
        if self.missing_info is None:
            self.missing_info = []

class ConfluenceAPI:
    """Confluence API 래퍼"""
    
    def __init__(self, url: str, username: str, api_token: str):
        self.confluence = Confluence(
            url=url,
            username=username,
            password=api_token,
            cloud=True
        )
    
    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """페이지 내용과 메타데이터 가져오기"""
        try:
            page_info = self.confluence.get_page_by_id(
                page_id=page_id,
                expand="body.storage,version,space"
            )
            
            html_content = page_info['body']['storage']['value']
            text_content = self._html_to_text(html_content)
            
            return {
                "id": page_id,
                "title": page_info['title'],
                "content": text_content,
                "html_content": html_content,
                "space_key": page_info['space']['key'],
                "version": page_info['version']['number'],
                "last_modified": page_info['version']['when'],
                "url": f"{self.confluence.url}/pages/viewpage.action?pageId={page_id}"
            }
            
        except Exception as e:
            logger.error(f"페이지 가져오기 실패: {e}")
            return {}
    
    def _html_to_text(self, html_content: str) -> str:
        """HTML을 텍스트로 변환"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for tag in soup(['script', 'style', 'meta']):
            tag.decompose()
        
        text_content = soup.get_text(separator='\n', strip=True)
        return text_content

class JiraAPI:
    """Jira API 래퍼"""
    
    def __init__(self, url: str, username: str, api_token: str):
        self.jira = Jira(
            url=url,
            username=username,
            password=api_token,
            cloud=True
        )
    
    def create_issue(self, requirement: RequirementItem, project_key: str, 
                    epic_key: Optional[str] = None) -> Dict[str, Any]:
        """이슈 생성"""
        fields = {
            "project": {"key": project_key},
            "summary": requirement.title,
            "description": self._format_description(requirement),
            "issuetype": {"name": requirement.category},
            "priority": {"name": requirement.priority}
        }
        
        if requirement.story_points and requirement.category in ["Story", "Task"]:
            fields["customfield_10016"] = requirement.story_points
        
        if requirement.labels:
            fields["labels"] = requirement.labels
        
        if requirement.components:
            fields["components"] = [{"name": comp} for comp in requirement.components]
        
        if epic_key and requirement.category != "Epic":
            fields["customfield_10014"] = epic_key
        
        try:
            issue = self.jira.issue_create(fields=fields)
            logger.info(f"이슈 생성 성공: {issue['key']}")
            return issue
        except Exception as e:
            logger.error(f"이슈 생성 실패: {e}")
            return {"error": str(e)}
    
    def create_epic(self, title: str, description: str, project_key: str) -> Dict[str, Any]:
        """에픽 생성"""
        fields = {
            "project": {"key": project_key},
            "summary": title,
            "description": description,
            "issuetype": {"name": "Epic"},
            "customfield_10011": title  # Epic Name
        }
        
        try:
            epic = self.jira.issue_create(fields=fields)
            logger.info(f"에픽 생성 성공: {epic['key']}")
            return epic
        except Exception as e:
            logger.error(f"에픽 생성 실패: {e}")
            return {"error": str(e)}
    
    def _format_description(self, requirement: RequirementItem) -> str:
        """Jira 설명 포맷팅"""
        description_parts = [requirement.description]
        
        if requirement.acceptance_criteria:
            description_parts.append("\n*Acceptance Criteria:*")
            for i, criteria in enumerate(requirement.acceptance_criteria, 1):
                description_parts.append(f"# {criteria}")
        
        if requirement.confidence_score < 0.8:
            description_parts.append(f"\n*Note: Confidence Score: {requirement.confidence_score:.2f} - May need further clarification*")
        
        return "\n".join(description_parts)

class RequirementAnalysisAgent:
    """LangGraph 기반 요구사항 분석 에이전트"""
    
    def __init__(self, confluence_config: Dict[str, str], jira_config: Dict[str, str], 
                 openai_api_key: str, model_name: str = "gpt-4"):
        
        # API 클라이언트 초기화
        self.confluence = ConfluenceAPI(**confluence_config)
        self.jira = JiraAPI(**jira_config)
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=0.1
        )
        
        # 체크포인터 (대화 상태 저장)
        self.checkpointer = SqliteSaver.from_conn_string(":memory:")
        
        # 워크플로우 그래프 구성
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
    
    def _build_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 구성"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("fetch_confluence", self.fetch_confluence_content)
        workflow.add_node("initial_analysis", self.initial_requirements_analysis)
        workflow.add_node("check_completeness", self.check_requirements_completeness)
        workflow.add_node("ask_clarification", self.ask_for_clarification)
        workflow.add_node("process_clarification", self.process_user_clarification)
        workflow.add_node("finalize_requirements", self.finalize_requirements)
        workflow.add_node("confirm_with_user", self.confirm_with_user)
        workflow.add_node("create_jira_tickets", self.create_jira_tickets)
        
        # 시작점 설정
        workflow.set_entry_point("fetch_confluence")
        
        # 엣지 정의
        workflow.add_edge("fetch_confluence", "initial_analysis")
        workflow.add_edge("initial_analysis", "check_completeness")
        
        # 조건부 엣지
        workflow.add_conditional_edges(
            "check_completeness",
            self.should_ask_clarification,
            {
                "ask_clarification": "ask_clarification",
                "finalize": "finalize_requirements"
            }
        )
        
        workflow.add_edge("ask_clarification", "process_clarification")
        workflow.add_edge("process_clarification", "check_completeness")
        workflow.add_edge("finalize_requirements", "confirm_with_user")
        
        workflow.add_conditional_edges(
            "confirm_with_user",
            self.should_create_tickets,
            {
                "create": "create_jira_tickets",
                "revise": "ask_clarification"
            }
        )
        
        workflow.add_edge("create_jira_tickets", END)
        
        return workflow
    
    async def fetch_confluence_content(self, state: AgentState) -> AgentState:
        """Confluence 페이지 내용 가져오기"""
        # 마지막 사용자 메시지에서 페이지 ID 추출
        last_message = state["messages"][-1]
        
        # 간단한 페이지 ID 추출 (실제로는 더 정교한 파싱 필요)
        import re
        page_id_match = re.search(r'\b\d{9,}\b', last_message.content)
        
        if not page_id_match:
            return {
                **state,
                "current_step": "error",
                "messages": state["messages"] + [
                    AIMessage(content="페이지 ID를 찾을 수 없습니다. 올바른 Confluence 페이지 ID를 제공해주세요.")
                ]
            }
        
        page_id = page_id_match.group()
        page_info = self.confluence.get_page_content(page_id)
        
        if not page_info:
            return {
                **state,
                "current_step": "error",
                "messages": state["messages"] + [
                    AIMessage(content=f"페이지 {page_id}를 가져올 수 없습니다. 페이지 ID와 권한을 확인해주세요.")
                ]
            }
        
        return {
            **state,
            "confluence_content": page_info["content"],
            "page_info": page_info,
            "current_step": "content_fetched",
            "messages": state["messages"] + [
                AIMessage(content=f"✅ 페이지 '{page_info['title']}'를 성공적으로 가져왔습니다. 요구사항 분석을 시작합니다...")
            ]
        }
    
    async def initial_requirements_analysis(self, state: AgentState) -> AgentState:
        """초기 요구사항 분석"""
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
당신은 소프트웨어 요구사항 분석 전문가입니다. 주어진 문서에서 개발 작업이 필요한 요구사항들을 식별하고 구조화해주세요.

각 요구사항에 대해 다음을 분석하세요:
1. 제목과 설명의 명확성
2. 우선순위와 복잡도
3. 인수기준의 구체성
4. 누락된 정보나 모호한 부분

분석 결과를 JSON 형식으로 반환하되, 각 요구사항에 대해 confidence_score(0-1)와 missing_info 필드를 포함해주세요.

JSON 형식:
{
    "requirements": [
        {
            "title": "간결하고 명확한 제목",
            "description": "상세한 설명",
            "priority": "Highest|High|Medium|Low|Lowest",
            "story_points": 1|2|3|5|8|13,
            "acceptance_criteria": ["구체적인 완료 조건들"],
            "category": "Epic|Story|Task|Bug",
            "labels": ["관련 태그들"],
            "components": ["관련 컴포넌트들"],
            "confidence_score": 0.85,
            "missing_info": ["부족한 정보 목록"]
        }
    ]
}
            """),
            HumanMessage(content=f"""
문서 제목: {state['page_info']['title']}

문서 내용:
{state['confluence_content']}

위 문서에서 소프트웨어 개발 요구사항들을 분석해주세요.
            """)
        ])
        
        try:
            response = await self.llm.ainvoke(analysis_prompt)
            
            # JSON 파싱
            json_start = response.content.find('{')
            json_end = response.content.rfind('}') + 1
            json_content = response.content[json_start:json_end]
            
            analysis_result = json.loads(json_content)
            requirements = analysis_result.get("requirements", [])
            
            return {
                **state,
                "raw_requirements": requirements,
                "current_step": "analyzed",
                "messages": state["messages"] + [
                    AIMessage(content=f"📊 초기 분석 완료: {len(requirements)}개의 요구사항을 발견했습니다.")
                ]
            }
            
        except Exception as e:
            logger.error(f"요구사항 분석 실패: {e}")
            return {
                **state,
                "current_step": "error",
                "messages": state["messages"] + [
                    AIMessage(content=f"요구사항 분석 중 오류가 발생했습니다: {str(e)}")
                ]
            }
    
    async def check_requirements_completeness(self, state: AgentState) -> AgentState:
        """요구사항 완성도 검사"""
        
        requirements = state.get("raw_requirements", [])
        needs_clarification = False
        clarification_questions = []
        
        # 사용자 피드백이 있는 경우 처리
        if state.get("user_clarifications"):
            # 사용자 답변을 바탕으로 요구사항 업데이트
            requirements = await self._update_requirements_with_clarifications(
                requirements, state["user_clarifications"]
            )
        
        # 완성도 검사
        for req in requirements:
            confidence = req.get("confidence_score", 0.0)
            missing_info = req.get("missing_info", [])
            
            if confidence < 0.7 or missing_info:
                needs_clarification = True
                
                # 구체적인 질문 생성
                for missing in missing_info:
                    if missing not in [q.split(":")[0] for q in clarification_questions]:
                        question = f"{req['title']}: {missing}에 대해 더 자세한 정보를 제공해주세요."
                        clarification_questions.append(question)
        
        # 일반적인 누락 정보 체크
        general_questions = self._generate_general_questions(requirements)
        clarification_questions.extend(general_questions)
        
        return {
            **state,
            "analyzed_requirements": requirements,
            "needs_clarification": needs_clarification,
            "clarification_questions": clarification_questions[:5],  # 최대 5개 질문
            "current_step": "completeness_checked"
        }
    
    def should_ask_clarification(self, state: AgentState) -> str:
        """클래리파이케이션이 필요한지 판단"""
        return "ask_clarification" if state.get("needs_clarification", False) else "finalize"
    
    async def ask_for_clarification(self, state: AgentState) -> AgentState:
        """사용자에게 추가 정보 요청"""
        
        questions = state.get("clarification_questions", [])
        
        if not questions:
            return {
                **state,
                "needs_clarification": False,
                "current_step": "no_clarification_needed"
            }
        
        question_text = "🤔 다음 사항들에 대해 추가 정보가 필요합니다:\n\n"
        for i, question in enumerate(questions, 1):
            question_text += f"{i}. {question}\n"
        
        question_text += "\n각 항목에 대해 답변해주세요. (예: 1. 상세 답변, 2. 상세 답변, ...)"
        
        return {
            **state,
            "current_step": "waiting_for_clarification",
            "messages": state["messages"] + [
                AIMessage(content=question_text)
            ]
        }
    
    async def process_user_clarification(self, state: AgentState) -> AgentState:
        """사용자 답변 처리"""
        
        # 마지막 사용자 메시지 가져오기
        last_user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_message = msg
                break
        
        if not last_user_message:
            return {
                **state,
                "current_step": "error",
                "messages": state["messages"] + [
                    AIMessage(content="사용자 답변을 찾을 수 없습니다.")
                ]
            }
        
        # 답변 파싱 (간단한 예제)
        clarifications = self._parse_user_clarifications(last_user_message.content)
        
        return {
            **state,
            "user_clarifications": clarifications,
            "current_step": "clarification_processed",
            "messages": state["messages"] + [
                AIMessage(content="✅ 추가 정보를 반영하여 요구사항을 업데이트하고 있습니다...")
            ]
        }
    
    async def finalize_requirements(self, state: AgentState) -> AgentState:
        """요구사항 최종화"""
        
        requirements = state.get("analyzed_requirements", [])
        
        # 최종 품질 검사 및 개선
        finalized_requirements = []
        for req in requirements:
            # 신뢰도가 높거나 사용자 피드백이 반영된 요구사항만 포함
            if req.get("confidence_score", 0) >= 0.7:
                finalized_requirements.append(req)
        
        return {
            **state,
            "final_requirements": finalized_requirements,
            "current_step": "finalized",
            "messages": state["messages"] + [
                AIMessage(content=f"✅ {len(finalized_requirements)}개의 요구사항이 최종 확정되었습니다.")
            ]
        }
    
    async def confirm_with_user(self, state: AgentState) -> AgentState:
        """사용자에게 최종 확인 요청"""
        
        requirements = state.get("final_requirements", [])
        
        confirmation_text = "📋 **최종 티켓 생성 확인**\n\n"
        confirmation_text += f"다음 {len(requirements)}개의 Jira 티켓을 생성할 예정입니다:\n\n"
        
        for i, req in enumerate(requirements, 1):
            confirmation_text += f"**{i}. {req['title']}**\n"
            confirmation_text += f"   - 타입: {req['category']}\n"
            confirmation_text += f"   - 우선순위: {req['priority']}\n"
            confirmation_text += f"   - 스토리 포인트: {req.get('story_points', 'N/A')}\n"
            confirmation_text += f"   - 설명: {req['description'][:100]}...\n\n"
        
        confirmation_text += "티켓을 생성하시겠습니까? (예/아니오로 답변해주세요)"
        
        return {
            **state,
            "current_step": "awaiting_confirmation",
            "messages": state["messages"] + [
                AIMessage(content=confirmation_text)
            ]
        }
    
    def should_create_tickets(self, state: AgentState) -> str:
        """티켓 생성 여부 판단"""
        
        # 마지막 사용자 메시지 확인
        last_user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_message = msg
                break
        
        if last_user_message:
            response = last_user_message.content.lower().strip()
            if any(word in response for word in ["예", "yes", "y", "네", "좋습니다", "생성"]):
                return "create"
        
        return "revise"
    
    async def create_jira_tickets(self, state: AgentState) -> AgentState:
        """Jira 티켓 생성"""
        
        requirements = state.get("final_requirements", [])
        page_info = state.get("page_info", {})
        
        # 프로젝트 키는 설정에서 가져오거나 사용자에게 요청 (여기서는 예제로 "PROJ" 사용)
        project_key = "PROJ"  # 실제로는 설정이나 사용자 입력에서 가져와야 함
        
        # 에픽 생성
        epic_title = f"[{page_info.get('title', 'Requirements')}] 구현"
        epic_description = f"Confluence 페이지 '{page_info.get('title', '')}' 요구사항 구현"
        
        epic_result = self.jira.create_epic(epic_title, epic_description, project_key)
        epic_key = epic_result.get('key') if 'error' not in epic_result else None
        
        # 개별 티켓 생성
        created_tickets = []
        failed_tickets = []
        
        for req_data in requirements:
            try:
                req = RequirementItem(**req_data)
                ticket_result = self.jira.create_issue(req, project_key, epic_key)
                
                if 'error' not in ticket_result:
                    created_tickets.append(ticket_result)
                else:
                    failed_tickets.append({"requirement": req.title, "error": ticket_result['error']})
                    
            except Exception as e:
                failed_tickets.append({"requirement": req_data.get('title', 'Unknown'), "error": str(e)})
        
        # 결과 메시지 생성
        result_text = f"🎉 **티켓 생성 완료!**\n\n"
        
        if epic_key:
            result_text += f"📈 **에픽**: {epic_key}\n\n"
        
        result_text += f"✅ **성공적으로 생성된 티켓**: {len(created_tickets)}개\n"
        for ticket in created_tickets:
            result_text += f"   - {ticket['key']}: {ticket.get('fields', {}).get('summary', 'Unknown')}\n"
        
        if failed_tickets:
            result_text += f"\n❌ **생성 실패한 티켓**: {len(failed_tickets)}개\n"
            for failed in failed_tickets:
                result_text += f"   - {failed['requirement']}: {failed['error']}\n"
        
        return {
            **state,
            "jira_tickets": created_tickets,
            "current_step": "completed",
            "messages": state["messages"] + [
                AIMessage(content=result_text)
            ]
        }
    
    async def _update_requirements_with_clarifications(self, requirements: List[Dict], 
                                                     clarifications: Dict[str, str]) -> List[Dict]:
        """사용자 피드백을 바탕으로 요구사항 업데이트"""
        
        update_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
사용자가 제공한 추가 정보를 바탕으로 요구사항들을 개선해주세요.
기존 요구사항의 confidence_score를 높이고, missing_info를 업데이트하며, 
필요한 경우 설명이나 인수기준을 보완해주세요.

동일한 JSON 형식으로 개선된 요구사항들을 반환해주세요.
            """),
            HumanMessage(content=f"""
기존 요구사항:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

사용자 추가 정보:
{json.dumps(clarifications, ensure_ascii=False, indent=2)}

요구사항을 개선해주세요.
            """)
        ])
        
        try:
            response = await self.llm.ainvoke(update_prompt)
            json_start = response.content.find('{')
            json_end = response.content.rfind('}') + 1
            json_content = response.content[json_start:json_end]
            
            updated_data = json.loads(json_content)
            return updated_data.get("requirements", requirements)
            
        except Exception as e:
            logger.error(f"요구사항 업데이트 실패: {e}")
            return requirements
    
    def _generate_general_questions(self, requirements: List[Dict]) -> List[str]:
        """일반적인 누락 정보에 대한 질문 생성"""
        
        questions = []
        
        # 전체적인 프로젝트 컨텍스트 확인
        if len(requirements) > 3:
            questions.append("이 요구사항들의 전체적인 우선순위나 일정이 있나요?")
        
        # 기술적 제약사항 확인
        questions.append("특별히 고려해야 할 기술적 제약사항이나 레거시 시스템 연동이 있나요?")
        
        # 사용자 그룹 확인
        questions.append("주요 사용자 그룹이나 페르소나에 대한 추가 정보가 있나요?")
        
        return questions
    
    def _parse_user_clarifications(self, user_input: str) -> Dict[str, str]:
        """사용자 답변 파싱 (간단한 예제)"""
        
        clarifications = {}
        
        # 번호가 있는 답변 형식 파싱 (1. 답변, 2. 답변, ...)
        import re
        numbered_answers = re.findall(r'(\d+)\.\s*([^\d]+?)(?=\d+\.|$)', user_input, re.DOTALL)
        
        for num, answer in numbered_answers:
            clarifications[f"question_{num}"] = answer.strip()
        
        # 번호가 없는 경우 전체 텍스트를 하나의 답변으로 처리
        if not clarifications:
            clarifications["general_clarification"] = user_input.strip()
        
        return clarifications
    
    async def run(self, initial_message: str, thread_id: str = "default") -> Dict[str, Any]:
        """에이전트 실행"""
        
        # 초기 상태 설정
        initial_state = {
            "messages": [HumanMessage(content=initial_message)],
            "confluence_content": None,
            "page_info": None,
            "raw_requirements": None,
            "analyzed_requirements": None,
            "user_clarifications": None,
            "final_requirements": None,
            "jira_tickets": None,
            "current_step": "started",
            "user_confirmed": False,
            "needs_clarification": False,
            "clarification_questions": None
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # 워크플로우 실행
        final_state = await self.app.ainvoke(initial_state, config=config)
        
        return {
            "status": "completed" if final_state["current_step"] == "completed" else "error",
            "final_state": final_state,
            "messages": [msg.content for msg in final_state["messages"]],
            "created_tickets": final_state.get("jira_tickets", []),
            "requirements_count": len(final_state.get("final_requirements", [])),
            "page_info": final_state.get("page_info", {})
        }

# 대화형 CLI 인터페이스
class InteractiveCLI:
    """대화형 명령줄 인터페이스"""
    
    def __init__(self, agent: RequirementAnalysisAgent):
        self.agent = agent
        self.thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_state = None
    
    async def start_conversation(self):
        """대화 시작"""
        print("🤖 Confluence 요구사항 분석 및 Jira 티켓 생성 에이전트")
        print("=" * 60)
        print("Confluence 페이지 ID를 입력하여 요구사항 분석을 시작하세요.")
        print("예: 123456789")
        print("종료하려면 'quit' 또는 'exit'을 입력하세요.\n")
        
        while True:
            try:
                user_input = input("👤 입력: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료']:
                    print("👋 에이전트를 종료합니다.")
                    break
                
                if not user_input:
                    continue
                
                print("🤔 처리 중...")
                
                # 첫 실행인지 아니면 대화 중인지 확인
                if self.conversation_state is None:
                    # 새로운 대화 시작
                    result = await self.agent.run(user_input, self.thread_id)
                    self.conversation_state = result["final_state"]
                else:
                    # 기존 대화 이어가기
                    current_state = self.conversation_state
                    current_state["messages"].append(HumanMessage(content=user_input))
                    
                    config = {"configurable": {"thread_id": self.thread_id}}
                    result = await self.agent.app.ainvoke(current_state, config=config)
                    self.conversation_state = result
                
                # 응답 출력
                self._print_agent_response(result)
                
                # 완료되면 세션 초기화
                if result.get("current_step") == "completed":
                    self.conversation_state = None
                    print("\n" + "=" * 60)
                    print("세션이 완료되었습니다. 새로운 페이지 ID를 입력하여 다시 시작할 수 있습니다.")
                    print("=" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 에이전트를 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류가 발생했습니다: {e}")
                print("다시 시도하거나 'quit'을 입력하여 종료하세요.\n")
    
    def _print_agent_response(self, result):
        """에이전트 응답 출력"""
        if isinstance(result, dict):
            if "messages" in result and result["messages"]:
                latest_ai_message = None
                for msg in reversed(result["messages"]):
                    if not msg.startswith("👤"):
                        latest_ai_message = msg
                        break
                
                if latest_ai_message:
                    print(f"🤖 {latest_ai_message}\n")
            else:
                print("🤖 처리가 완료되었습니다.\n")
        else:
            print(f"🤖 {result}\n")

# 웹 인터페이스 (FastAPI 사용)
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

class ChatRequest(BaseModel):
    message: str
    thread_id: str = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    status: str
    step: str

app = FastAPI(title="Requirements Analysis Agent API")

# 글로벌 에이전트 인스턴스 (실제 환경에서는 더 나은 방법 사용)
global_agent = None

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """간단한 웹 채팅 인터페이스"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Requirements Analysis Agent</title>
        <meta charset="UTF-8">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 20px; 
                background-color: #f5f5f5;
            }
            .chat-container { 
                background: white; 
                border-radius: 10px; 
                padding: 20px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .messages { 
                height: 400px; 
                overflow-y: auto; 
                border: 1px solid #ddd; 
                padding: 15px; 
                margin-bottom: 15px; 
                background: #fafafa;
                border-radius: 5px;
            }
            .message { 
                margin-bottom: 15px; 
                padding: 10px; 
                border-radius: 5px;
            }
            .user { 
                background: #007bff; 
                color: white; 
                text-align: right; 
            }
            .agent { 
                background: #e9ecef; 
                color: #333; 
            }
            .input-group { 
                display: flex; 
                gap: 10px; 
            }
            input { 
                flex: 1; 
                padding: 10px; 
                border: 1px solid #ddd; 
                border-radius: 5px;
                font-size: 16px;
            }
            button { 
                padding: 10px 20px; 
                background: #007bff; 
                color: white; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer;
                font-size: 16px;
            }
            button:hover { 
                background: #0056b3; 
            }
            .status { 
                margin-top: 10px; 
                padding: 10px; 
                border-radius: 5px; 
                font-weight: bold;
            }
            .status.processing { 
                background: #fff3cd; 
                color: #856404; 
            }
            .status.completed { 
                background: #d4edda; 
                color: #155724; 
            }
            .status.error { 
                background: #f8d7da; 
                color: #721c24; 
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <h1>🤖 Requirements Analysis Agent</h1>
            <p>Confluence 페이지 ID를 입력하여 요구사항 분석을 시작하세요.</p>
            
            <div id="messages" class="messages">
                <div class="message agent">
                    👋 안녕하세요! Confluence 페이지 ID를 입력해주세요. (예: 123456789)
                </div>
            </div>
            
            <div class="input-group">
                <input type="text" id="messageInput" placeholder="메시지를 입력하세요..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()">전송</button>
            </div>
            
            <div id="status" class="status" style="display: none;"></div>
        </div>

        <script>
            let threadId = 'session_' + Date.now();
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (!message) return;
                
                // 사용자 메시지 표시
                addMessage(message, 'user');
                input.value = '';
                
                // 상태 표시
                showStatus('처리 중...', 'processing');
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            thread_id: threadId
                        })
                    });
                    
                    const data = await response.json();
                    
                    // 에이전트 응답 표시
                    addMessage(data.response, 'agent');
                    
                    // 상태 업데이트
                    showStatus(`현재 단계: ${data.step}`, data.status);
                    
                } catch (error) {
                    console.error('Error:', error);
                    addMessage('오류가 발생했습니다. 다시 시도해주세요.', 'agent');
                    showStatus('오류 발생', 'error');
                }
            }
            
            function addMessage(text, type) {
                const messages = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}`;
                messageDiv.innerHTML = text.replace(/\n/g, '<br>');
                messages.appendChild(messageDiv);
                messages.scrollTop = messages.scrollHeight;
            }
            
            function showStatus(text, type) {
                const status = document.getElementById('status');
                status.textContent = text;
                status.className = `status ${type}`;
                status.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """채팅 API 엔드포인트"""
    if global_agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        thread_id = request.thread_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 에이전트 실행
        result = await global_agent.run(request.message, thread_id)
        
        # 최신 AI 메시지 추출
        latest_ai_message = ""
        if "messages" in result["final_state"] and result["final_state"]["messages"]:
            for msg in reversed(result["final_state"]["messages"]):
                if isinstance(msg, AIMessage):
                    latest_ai_message = msg.content
                    break
        
        return ChatResponse(
            response=latest_ai_message or "처리 완료",
            thread_id=thread_id,
            status=result["status"],
            step=result["final_state"].get("current_step", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Chat API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/initialize")
async def initialize_agent(
    confluence_url: str,
    confluence_username: str, 
    confluence_token: str,
    jira_url: str,
    jira_username: str,
    jira_token: str,
    openai_api_key: str,
    model_name: str = "gpt-4"
):
    """에이전트 초기화"""
    global global_agent
    
    try:
        confluence_config = {
            "url": confluence_url,
            "username": confluence_username,
            "api_token": confluence_token
        }
        
        jira_config = {
            "url": jira_url,
            "username": jira_username, 
            "api_token": jira_token
        }
        
        global_agent = RequirementAnalysisAgent(
            confluence_config=confluence_config,
            jira_config=jira_config,
            openai_api_key=openai_api_key,
            model_name=model_name
        )
        
        return {"status": "success", "message": "Agent initialized successfully"}
        
    except Exception as e:
        logger.error(f"에이전트 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 메인 실행 함수들
async def main_interactive():
    """대화형 CLI 실행"""
    
    # 설정 로드
    config = load_config()
    
    # 에이전트 초기화
    agent = RequirementAnalysisAgent(
        confluence_config=config["confluence"],
        jira_config=config["jira"], 
        openai_api_key=config["openai_api_key"],
        model_name=config.get("model_name", "gpt-4")
    )
    
    # 대화형 CLI 시작
    cli = InteractiveCLI(agent)
    await cli.start_conversation()

def main_web():
    """웹 인터페이스 실행"""
    
    # 설정 로드
    config = load_config()
    
    # 글로벌 에이전트 초기화
    global global_agent
    global_agent = RequirementAnalysisAgent(
        confluence_config=config["confluence"],
        jira_config=config["jira"],
        openai_api_key=config["openai_api_key"], 
        model_name=config.get("model_name", "gpt-4")
    )
    
    print("🚀 웹 인터페이스를 시작합니다...")
    print("브라우저에서 http://localhost:8000 에 접속하세요.")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

async def main_single_run():
    """단일 실행 예제"""
    
    config = load_config()
    
    agent = RequirementAnalysisAgent(
        confluence_config=config["confluence"],
        jira_config=config["jira"],
        openai_api_key=config["openai_api_key"],
        model_name=config.get("model_name", "gpt-4")
    )
    
    # 페이지 ID 입력 받기
    page_id = input("Confluence 페이지 ID를 입력하세요: ").strip()
    
    if not page_id:
        print("페이지 ID가 필요합니다.")
        return
    
    print("🤖 요구사항 분석을 시작합니다...")
    
    result = await agent.run(f"페이지 ID: {page_id}")
    
    print("\n" + "=" * 60)
    print("🎉 분석 완료!")
    print("=" * 60)
    
    for i, message in enumerate(result["messages"]):
        if i % 2 == 0:
            print(f"👤 {message}")
        else:
            print(f"🤖 {message}")
        print()

def load_config() -> Dict[str, Any]:
    """설정 파일 로드"""
    
    import os
    
    # 환경 변수에서 설정 로드
    config = {
        "confluence": {
            "url": os.getenv("CONFLUENCE_URL", "https://your-domain.atlassian.net/wiki"),
            "username": os.getenv("CONFLUENCE_USERNAME", "your-email@example.com"),
            "api_token": os.getenv("CONFLUENCE_TOKEN", "your-confluence-token")
        },
        "jira": {
            "url": os.getenv("JIRA_URL", "https://your-domain.atlassian.net"),
            "username": os.getenv("JIRA_USERNAME", "your-email@example.com"),
            "api_token": os.getenv("JIRA_TOKEN", "your-jira-token")
        },
        "openai_api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key"),
        "model_name": os.getenv("OPENAI_MODEL", "gpt-4")
    }
    
    # config.json 파일이 있으면 우선 적용
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            file_config = json.load(f)
            config.update(file_config)
    except FileNotFoundError:
        pass
    
    return config

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "web":
            main_web()
        elif mode == "single":
            asyncio.run(main_single_run())
        elif mode == "interactive" or mode == "cli":
            asyncio.run(main_interactive())
        else:
            print("사용법:")
            print("  python agent.py interactive  # 대화형 CLI")
            print("  python agent.py web         # 웹 인터페이스")
            print("  python agent.py single      # 단일 실행")
    else:
        # 기본값: 대화형 CLI
        asyncio.run(main_interactive())
