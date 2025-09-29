# server/main.py
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import uvicorn
import asyncio
import json
from datetime import datetime

# Import components
from orchestrator import AgentOrchestrator
from model_adapter import ModelAdapter
from kce import KnowledgeContextEngine
from config import Settings

app = FastAPI(title="IoT AI Agent API")
settings = Settings()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 컴포넌트 초기화
model_adapter = ModelAdapter(settings)
kce = KnowledgeContextEngine(settings)
orchestrator = AgentOrchestrator(model_adapter, kce, settings)

# Request/Response Models
class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    mode: str = "on_demand"  # on_demand or proactive

class ChatResponse(BaseModel):
    session_id: str
    message: str
    actions: Optional[List[Dict[str, Any]]] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: str

class SessionState:
    def __init__(self):
        self.sessions = {}
    
    def get_or_create(self, session_id: str, user_id: str) -> Dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "user_id": user_id,
                "history": [],
                "context": {},
                "created_at": datetime.now().isoformat()
            }
        return self.sessions[session_id]
    
    def update(self, session_id: str, data: Dict):
        if session_id in self.sessions:
            self.sessions[session_id].update(data)
    
    def delete(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

session_state = SessionState()

@app.get("/")
async def root():
    return {"message": "IoT AI Agent API", "version": "1.0.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """처리 on-demand mode 대화"""
    try:
        # 세션 관리
        session_id = request.session_id or f"{request.user_id}_{datetime.now().timestamp()}"
        session = session_state.get_or_create(session_id, request.user_id)
        
        # Orchestrator로 메시지 처리
        result = await orchestrator.process_message(
            user_id=request.user_id,
            message=request.message,
            session=session,
            context=request.context,
            mode=request.mode
        )
        
        # 세션 업데이트
        session["history"].append({
            "user": request.message,
            "assistant": result["message"],
            "timestamp": datetime.now().isoformat()
        })
        session_state.update(session_id, session)
        
        return ChatResponse(
            session_id=session_id,
            message=result["message"],
            actions=result.get("actions"),
            context=result.get("context"),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket for streaming responses"""
    await websocket.accept()
    session_id = f"{user_id}_{datetime.now().timestamp()}"
    session = session_state.get_or_create(session_id, user_id)
    
    try:
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # 스트리밍 응답 처리
            async for chunk in orchestrator.process_message_stream(
                user_id=user_id,
                message=message_data["message"],
                session=session,
                context=message_data.get("context"),
                mode=message_data.get("mode", "on_demand")
            ):
                await websocket.send_text(json.dumps(chunk))
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
    finally:
        await websocket.close()

@app.post("/session/{session_id}/reset")
async def reset_session(session_id: str):
    """세션 초기화"""
    session_state.delete(session_id)
    return {"message": "Session reset", "session_id": session_id}

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "components": {
            "model_adapter": model_adapter.health_check(),
            "kce": kce.health_check(),
            "orchestrator": orchestrator.health_check()
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)