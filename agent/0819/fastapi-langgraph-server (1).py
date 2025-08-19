# FastAPI + LangGraph Server
# pip install fastapi uvicorn langgraph langchain langchain-community langchain-ollama python-multipart

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, AsyncGenerator
import json
import asyncio
import uuid
from datetime import datetime
import os
from pathlib import Path

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import Annotated
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LangGraph Multi-Agent API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    agent_type: str
    model: str = "qwen2.5:8b"
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    agent_state: str
    metadata: Dict[str, Any] = {}

class ConversationState(BaseModel):
    messages: List[ChatMessage] = []
    agent_type: str
    current_state: str
    metadata: Dict[str, Any] = {}

# Global variables
conversations: Dict[str, ConversationState] = {}
vectorstores: Dict[str, FAISS] = {}
ollama_client = None
embeddings = None

# Initialize Ollama client
def init_ollama(model_name: str = "qwen2.5:8b", base_url: str = "http://localhost:11434"):
    global ollama_client, embeddings
    try:
        ollama_client = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.7,
        )
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",  # 임베딩용 모델
            base_url=base_url
        )
        logger.info(f"Ollama client initialized with model: {model_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Ollama: {e}")
        return False

# Agent State Definition
class AgentState(BaseModel):
    messages: Annotated[List[Any], add_messages]
    current_node: str = "start"
    agent_type: str
    context: str = ""
    user_info: Dict[str, Any] = {}

# Agent System Prompts
AGENT_PROMPTS = {
    "doctor": """당신은 전문 의료진 AI입니다. 환자의 증상을 듣고 정확하고 도움이 되는 의료 조언을 제공합니다.
주의사항:
- 정확한 진단은 실제 의사의 진료가 필요함을 강조
- 응급상황에서는 즉시 병원 방문을 권유
- 전문적이지만 이해하기 쉬운 용어 사용
- 한국어로 친근하게 대화
- 현재 단계: {current_state}에 맞는 적절한 질문과 조언 제공""",
    
    "travel": """당신은 전문 여행 상담사 AI입니다. 고객의 여행 계획을 도와드립니다.
서비스 범위:
- 항공편 검색 및 추천
- 호텔 및 숙박 시설 추천
- 렌터카 및 교통편 안내
- 여행 일정 계획
- 현지 정보 및 팁 제공
- 현재 단계: {current_state}에 맞는 구체적인 정보 수집 및 제안
한국어로 친절하고 상세하게 안내해주세요.""",
    
    "movie": """당신은 영화 예매 전문 AI입니다. 고객의 영화 관람을 도와드립니다.
서비스 범위:
- 현재 상영작 추천
- 상영 시간표 안내
- 좌석 선택 도움
- 예매 과정 안내
- 영화 정보 및 리뷰 제공
- 현재 단계: {current_state}에 맞는 단계별 안내
한국어로 친근하고 도움이 되게 대화해주세요."""
}

# Agent Graph Definitions
AGENT_GRAPHS = {
    "doctor": {
        "nodes": ["초기진단", "증상분석", "검사제안", "처방제안", "추가상담"],
        "edges": [
            ("초기진단", "증상분석"),
            ("증상분석", "검사제안"),
            ("증상분석", "처방제안"),
            ("검사제안", "추가상담"),
            ("처방제안", "추가상담")
        ]
    },
    "travel": {
        "nodes": ["여행계획", "항공검색", "호텔검색", "렌터카", "일정조정", "예약확정"],
        "edges": [
            ("여행계획", "항공검색"),
            ("여행계획", "호텔검색"),
            ("항공검색", "일정조정"),
            ("호텔검색", "일정조정"),
            ("일정조정", "렌터카"),
            ("렌터카", "예약확정")
        ]
    },
    "movie": {
        "nodes": ["영화검색", "상영시간", "좌석선택", "결제처리", "예약완료"],
        "edges": [
            ("영화검색", "상영시간"),
            ("상영시간", "좌석선택"),
            ("좌석선택", "결제처리"),
            ("결제처리", "예약완료")
        ]
    }
}

class MultiAgentSystem:
    def __init__(self):
        self.graphs = {}
        self._build_graphs()
    
    def _build_graphs(self):
        """Build LangGraph for each agent type"""
        for agent_type in AGENT_GRAPHS.keys():
            self.graphs[agent_type] = self._create_agent_graph(agent_type)
    
    def _create_agent_graph(self, agent_type: str):
        """Create a LangGraph for specific agent"""
        graph = StateGraph(AgentState)
        
        # Add nodes
        nodes = AGENT_GRAPHS[agent_type]["nodes"]
        for node in nodes:
            graph.add_node(node, self._create_node_function(agent_type, node))
        
        # Add edges
        edges = AGENT_GRAPHS[agent_type]["edges"]
        for from_node, to_node in edges:
            graph.add_edge(from_node, to_node)
        
        # Set entry point
        graph.add_edge(START, nodes[0])
        graph.add_edge(nodes[-1], END)
        
        return graph.compile()
    
    def _create_node_function(self, agent_type: str, node_name: str):
        """Create a function for each node"""
        async def node_function(state: AgentState):
            # Update current node
            state.current_node = node_name
            
            # Get the last human message
            human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
            if not human_messages:
                return state
            
            last_message = human_messages[-1].content
            
            # Create system prompt with current state
            system_prompt = AGENT_PROMPTS[agent_type].format(current_state=node_name)
            
            # Add context if available
            context_prompt = ""
            if state.context:
                context_prompt = f"\n\n참고 정보:\n{state.context}"
            
            # Create messages for LLM
            messages = [
                SystemMessage(content=system_prompt + context_prompt),
                HumanMessage(content=last_message)
            ]
            
            try:
                # Generate response using Ollama
                response = await ollama_client.ainvoke(messages)
                state.messages.append(AIMessage(content=response.content))
                
                # Determine next state based on conversation flow
                state = self._determine_next_state(state, agent_type, node_name)
                
            except Exception as e:
                logger.error(f"Error in node {node_name}: {e}")
                error_msg = f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
                state.messages.append(AIMessage(content=error_msg))
            
            return state
        
        return node_function
    
    def _determine_next_state(self, state: AgentState, agent_type: str, current_node: str):
        """Determine next state based on conversation context"""
        edges = AGENT_GRAPHS[agent_type]["edges"]
        possible_next = [to_node for from_node, to_node in edges if from_node == current_node]
        
        if possible_next:
            # Simple logic: move to first possible next state
            # In a more sophisticated system, this could use NLP to determine the best next state
            state.current_node = possible_next[0]
        
        return state

# Initialize the multi-agent system
agent_system = MultiAgentSystem()

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    success = init_ollama()
    if not success:
        logger.warning("Failed to initialize Ollama. Some features may not work.")

@app.get("/")
async def root():
    return {"message": "LangGraph Multi-Agent API Server", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Ollama connection
        test_message = [HumanMessage(content="Hello")]
        response = await ollama_client.ainvoke(test_message)
        ollama_status = "connected"
    except:
        ollama_status = "disconnected"
    
    return {
        "status": "healthy",
        "ollama": ollama_status,
        "conversations": len(conversations),
        "vectorstores": len(vectorstores)
    }

@app.get("/models")
async def get_available_models():
    """Get available Ollama models"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return {"models": models}
        else:
            return {"models": ["qwen2.5:8b"]}  # fallback
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        return {"models": ["qwen2.5:8b"]}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        # Get or create conversation
        if request.conversation_id not in conversations:
            conversations[request.conversation_id] = ConversationState(
                agent_type=request.agent_type,
                current_state=AGENT_GRAPHS[request.agent_type]["nodes"][0]
            )
        
        conversation = conversations[request.conversation_id]
        
        # Add user message
        user_message = ChatMessage(
            role="user", 
            content=request.message,
            timestamp=datetime.now()
        )
        conversation.messages.append(user_message)
        
        # Get RAG context if available
        context = ""
        if request.conversation_id in vectorstores:
            context = await get_rag_context(request.conversation_id, request.message)
        
        # Create agent state
        agent_state = AgentState(
            messages=[HumanMessage(content=request.message)],
            agent_type=request.agent_type,
            context=context,
            current_node=conversation.current_state
        )
        
        # Run the agent graph
        graph = agent_system.graphs[request.agent_type]
        result = await graph.ainvoke(agent_state)
        
        # Extract AI response
        ai_messages = [msg for msg in result.messages if isinstance(msg, AIMessage)]
        response_content = ai_messages[-1].content if ai_messages else "응답을 생성할 수 없습니다."
        
        # Update conversation
        ai_message = ChatMessage(
            role="assistant",
            content=response_content,
            timestamp=datetime.now()
        )
        conversation.messages.append(ai_message)
        conversation.current_state = result.current_node
        
        return ChatResponse(
            response=response_content,
            conversation_id=request.conversation_id,
            agent_state=result.current_node,
            metadata={"model": request.model}
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint"""
    async def generate():
        try:
            # Similar setup as chat endpoint
            if request.conversation_id not in conversations:
                conversations[request.conversation_id] = ConversationState(
                    agent_type=request.agent_type,
                    current_state=AGENT_GRAPHS[request.agent_type]["nodes"][0]
                )
            
            conversation = conversations[request.conversation_id]
            
            # Add user message
            user_message = ChatMessage(
                role="user", 
                content=request.message,
                timestamp=datetime.now()
            )
            conversation.messages.append(user_message)
            
            # For streaming, we'll simulate token-by-token response
            # In a real implementation, you'd use Ollama's streaming API
            response_text = f"[{conversation.current_state}] 스트리밍 응답 예시: {request.message}에 대한 답변입니다."
            
            for i, char in enumerate(response_text):
                chunk = {
                    "content": char,
                    "conversation_id": request.conversation_id,
                    "agent_state": conversation.current_state,
                    "is_complete": i == len(response_text) - 1
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.05)  # Simulate streaming delay
                
        except Exception as e:
            error_chunk = {
                "error": str(e),
                "conversation_id": request.conversation_id
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/upload")
async def upload_file(conversation_id: str, file: UploadFile = File(...)):
    """Upload and process RAG documents"""
    try:
        if not file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="Only .txt files are supported")
        
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{conversation_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process for RAG
        await process_rag_document(conversation_id, file_path)
        
        return {"message": "File uploaded and processed successfully", "filename": file.filename}
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_rag_document(conversation_id: str, file_path: Path):
    """Process document for RAG"""
    try:
        # Load and split document
        loader = TextLoader(str(file_path), encoding='utf-8')
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstores[conversation_id] = vectorstore
        
        logger.info(f"RAG document processed for conversation {conversation_id}")
        
    except Exception as e:
        logger.error(f"RAG processing error: {e}")
        raise

async def get_rag_context(conversation_id: str, query: str, k: int = 3) -> str:
    """Get relevant context from RAG"""
    try:
        if conversation_id not in vectorstores:
            return ""
        
        vectorstore = vectorstores[conversation_id]
        docs = vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
        
    except Exception as e:
        logger.error(f"RAG context error: {e}")
        return ""

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversations[conversation_id]

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete conversation and associated data"""
    if conversation_id in conversations:
        del conversations[conversation_id]
    
    if conversation_id in vectorstores:
        del vectorstores[conversation_id]
    
    # Clean up uploaded files
    upload_dir = Path("uploads")
    if upload_dir.exists():
        for file in upload_dir.glob(f"{conversation_id}_*"):
            file.unlink()
    
    return {"message": "Conversation deleted successfully"}

@app.get("/agent-graph/{agent_type}")
async def get_agent_graph(agent_type: str):
    """Get agent graph structure"""
    if agent_type not in AGENT_GRAPHS:
        raise HTTPException(status_code=404, detail="Agent type not found")
    
    return AGENT_GRAPHS[agent_type]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)