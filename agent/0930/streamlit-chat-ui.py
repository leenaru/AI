# client/streamlit_app.py
import streamlit as st
import requests
import json
import asyncio
import websocket
from datetime import datetime
import uuid
from typing import Dict, List, Optional
import base64
from PIL import Image
import io

# 페이지 설정
st.set_page_config(
    page_title="IoT AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API 설정
API_BASE_URL = st.secrets.get("API_URL", "http://localhost:8000")

# CSS 스타일
def load_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from { 
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.95);
        margin-right: 20%;
    }
    
    .action-card {
        background: rgba(255, 255, 255, 0.9);
        border: 2px dashed #667eea;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .status-online {
        background-color: #10b981;
    }
    
    .status-offline {
        background-color: #ef4444;
    }
    
    .status-processing {
        background-color: #f59e0b;
    }
    </style>
    """, unsafe_allow_html=True)

# 세션 상태 초기화
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"
    if "context" not in st.session_state:
        st.session_state.context = {}
    if "mode" not in st.session_state:
        st.session_state.mode = "on_demand"
    if "api_status" not in st.session_state:
        st.session_state.api_status = "checking"

# API 상태 체크
def check_api_status():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            return "online", response.json()
        else:
            return "offline", None
    except:
        return "offline", None

# 메시지 전송
def send_message(message: str, context: Optional[Dict] = None):
    """API로 메시지 전송"""
    try:
        payload = {
            "user_id": st.session_state.user_id,
            "message": message,
            "session_id": st.session_state.session_id,
            "context": context or st.session_state.context,
            "mode": st.session_state.mode
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# 액션 처리
def process_action(action: Dict):
    """서버에서 받은 액션 처리"""
    action_type = action.get("type")
    
    if action_type == "camera_scan":
        st.info(f"📷 {action.get('description', '카메라를 준비해주세요')}")
        # 실제 앱에서는 카메라 활성화
        uploaded_file = st.file_uploader(
            "이미지를 업로드하세요",
            type=["jpg", "jpeg", "png"],
            key=f"camera_{datetime.now().timestamp()}"
        )
        if uploaded_file:
            process_image(uploaded_file)
            
    elif action_type == "show_menu":
        st.write("📋 **메뉴 옵션:**")
        options = action.get("options", [])
        for opt in options:
            if st.button(opt, key=f"menu_{opt}"):
                send_message(f"메뉴 선택: {opt}")
                
    elif action_type == "start_onboarding":
        device = action.get("device")
        steps = action.get("steps", [])
        st.success(f"✅ {device} 온보딩을 시작합니다")
        st.write(f"단계: {', '.join(steps)}")
        
    elif action_type == "voc_submit":
        ticket_id = action.get("ticket_id")
        st.success(f"📝 VOC 접수 완료: {ticket_id}")

def process_image(uploaded_file):
    """이미지 처리 (시뮬레이션)"""
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 이미지", use_column_width=True)
    
    # 이미지 분석 시뮬레이션
    with st.spinner("이미지 분석 중..."):
        # 실제로는 Vision API나 on-device 모델 사용
        st.session_state.context["image_analyzed"] = True
        st.session_state.context["detected_device"] = "Smart Hub Model X"

# 메인 UI
def main():
    load_css()
    init_session_state()
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1 style="text-align: center; color: #667eea;">
            🤖 IoT AI Assistant
        </h1>
        <p style="text-align: center; color: #666;">
            IoT 기기의 설치부터 장애까지, AI가 도와드립니다
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.title("⚙️ 설정")
        
        # API 상태
        status, health_data = check_api_status()
        st.session_state.api_status = status
        
        status_color = "status-online" if status == "online" else "status-offline"
        st.markdown(f"""
        <div>
            <span class="status-indicator {status_color}"></span>
            <span>서버 상태: {status.upper()}</span>
        </div>
        """, unsafe_allow_html=True)
        
        if health_data:
            with st.expander("상세 상태"):
                st.json(health_data)
        
        st.divider()
        
        # 모드 선택
        st.session_state.mode = st.selectbox(
            "대화 모드",
            ["on_demand", "proactive"],
            index=0,
            format_func=lambda x: "On-Demand (질문응답)" if x == "on_demand" else "Proactive (자동감지)"
        )
        
        # 세션 정보
        st.divider()
        st.subheader("📊 세션 정보")
        st.text(f"User ID: {st.session_state.user_id}")
        st.text(f"Session: {st.session_state.session_id[:8]}...")
        st.text(f"메시지 수: {len(st.session_state.messages)}")
        
        # 세션 리셋
        if st.button("🔄 새 대화 시작", type="secondary"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.context = {}
            st.rerun()
        
        # 빠른 명령
        st.divider()
        st.subheader("⚡ 빠른 명령")
        
        quick_commands = {
            "기기 등록": "새로운 스마트홈 기기를 등록하고 싶어요",
            "문제 해결": "WiFi 연결이 계속 끊겨요",
            "에러 코드": "E002 에러가 나타났어요",
            "사용 방법": "스마트 조명 밝기 조절은 어떻게 하나요?",
            "VOC 접수": "제품에 문제가 있어서 불만을 접수하고 싶어요"
        }
        
        for label, command in quick_commands.items():
            if st.button(label, key=f"quick_{label}"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": command,
                    "timestamp": datetime.now()
                })
                
                with st.spinner("처리 중..."):
                    response = send_message(command)
                    if "error" not in response:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.get("message", ""),
                            "actions": response.get("actions", []),
                            "timestamp": datetime.now()
                        })
                st.rerun()
    
    # 메인 채팅 영역
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 채팅 히스토리
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>
                        {content}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant:</strong><br>
                        {content}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 액션 표시
                    if message.get("actions"):
                        for action in message["actions"]:
                            st.markdown("""
                            <div class="action-card">
                            """, unsafe_allow_html=True)
                            process_action(action)
                            st.markdown("</div>", unsafe_allow_html=True)
        
        # 입력 영역
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([5, 1])
            
            with col_input:
                user_input = st.text_input(
                    "메시지 입력",
                    placeholder="무엇을 도와드릴까요?",
                    label_visibility="collapsed"
                )
            
            with col_send:
                submitted = st.form_submit_button("전송 ➤", use_container_width=True)
            
            if submitted and user_input:
                # 사용자 메시지 추가
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now()
                })
                
                # API 호출
                with st.spinner("AI가 답변을 준비하고 있습니다..."):
                    response = send_message(user_input)
                    
                    if "error" in response:
                        st.error(f"오류: {response['error']}")
                    else:
                        # 응답 추가
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.get("message", "죄송합니다. 응답을 생성할 수 없습니다."),
                            "actions": response.get("actions", []),
                            "timestamp": datetime.now()
                        })
                        
                        # 컨텍스트 업데이트
                        if response.get("context"):
                            st.session_state.context.update(response["context"])
                
                st.rerun()
    
    with col2:
        st.subheader("🎯 현재 컨텍스트")
        
        if st.session_state.context:
            for key, value in st.session_state.context.items():
                st.write(f"**{key}:** {value}")
        else:
            st.write("_컨텍스트 없음_")
        
        st.divider()
        
        # 도움말
        with st.expander("💡 도움말"):
            st.markdown("""
            **사용 가능한 기능:**
            - 🔧 기기 온보딩
            - 🔍 문제 해결
            - 📝 VOC 접수
            - ❓ 에러 코드 해결
            - 📖 매뉴얼 조회
            - 🛒 구매 가이드
            
            **팁:**
            - 구체적으로 질문하면 더 정확한 답변을 받을 수 있습니다
            - 이미지를 업로드하여 기기를 인식시킬 수 있습니다
            - 에러 코드는 정확하게 입력해주세요
            """)

if __name__ == "__main__":
    main()