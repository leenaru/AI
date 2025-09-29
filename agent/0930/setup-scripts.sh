#!/bin/bash
# setup.sh - 초기 설정 스크립트

echo "🚀 IoT AI Agent 설정을 시작합니다..."

# 디렉토리 구조 생성
echo "📁 디렉토리 구조 생성 중..."
mkdir -p server client logs data/models data/indices

# 가상환경 생성
echo "🐍 Python 가상환경 생성 중..."
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
echo "📦 Python 패키지 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt

# Ollama 모델 다운로드
echo "🤖 Ollama 모델 다운로드 중..."
if command -v ollama &> /dev/null; then
    ollama pull gemma2:2b
    ollama pull llama3.2:1b
else
    echo "⚠️  Ollama가 설치되지 않았습니다. https://ollama.ai 에서 설치해주세요."
fi

# 환경 변수 파일 생성
if [ ! -f .env ]; then
    echo "⚙️  환경 변수 파일 생성 중..."
    cat > .env <<EOL
# Model Settings
MODEL_BACKEND=OLLAMA
MODEL_NAME=gemma2:2b
EMBEDDING_DIM=768

# API Keys (Optional)
OPENAI_API_KEY=
HUGGING_FACE_HUB_TOKEN=
LANGSMITH_API_KEY=

# Server Settings
LOG_LEVEL=INFO
DEBUG=False

# URLs
OLLAMA_BASE_URL=http://localhost:11434
VLLM_BASE_URL=http://localhost:8001
REDIS_URL=redis://localhost:6379
EOL
fi

echo "✅ 설정 완료!"

---
#!/bin/bash
# start.sh - 시스템 시작 스크립트

echo "🚀 IoT AI Agent 시스템을 시작합니다..."

# Docker Compose로 시작하는 경우
if [ "$1" == "docker" ]; then
    echo "🐳 Docker Compose로 시작 중..."
    docker-compose up -d
    echo "✅ 시스템이 시작되었습니다!"
    echo "📍 API Server: http://localhost:8000"
    echo "📍 Streamlit UI: http://localhost:8501"
    echo "📍 API Docs: http://localhost:8000/docs"
    exit 0
fi

# 로컬 개발 환경으로 시작
echo "💻 로컬 개발 환경으로 시작 중..."

# 가상환경 활성화
source venv/bin/activate

# Ollama 시작 (백그라운드)
if command -v ollama &> /dev/null; then
    echo "🤖 Ollama 서버 시작 중..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 5
else
    echo "⚠️  Ollama가 실행 중이 아닙니다. 수동으로 시작해주세요."
fi

# Redis 시작 (Docker)
echo "📦 Redis 시작 중..."
docker run -d --name iot-redis -p 6379:6379 redis:alpine

# FastAPI 서버 시작
echo "🔧 API 서버 시작 중..."
cd server
uvicorn main:app --reload --port 8000 &
API_PID=$!
cd ..

# Streamlit UI 시작
echo "🎨 Streamlit UI 시작 중..."
cd client
streamlit run streamlit_app.py --server.port 8501 &
UI_PID=$!
cd ..

echo "✅ 시스템이 시작되었습니다!"
echo "📍 API Server: http://localhost:8000"
echo "📍 Streamlit UI: http://localhost:8501"
echo "📍 API Docs: http://localhost:8000/docs"
echo ""
echo "종료하려면 Ctrl+C를 누르세요."

# 종료 처리
trap "kill $OLLAMA_PID $API_PID $UI_PID; docker stop iot-redis; docker rm iot-redis" EXIT

# 프로세스 대기
wait

---
#!/bin/bash
# test.sh - 테스트 스크립트

echo "🧪 시스템 테스트를 시작합니다..."

# API 헬스체크
echo "1️⃣ API 서버 상태 확인..."
curl -s http://localhost:8000/health | python3 -m json.tool

echo ""
echo "2️⃣ 샘플 채팅 테스트..."
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "message": "스마트 허브를 설정하고 싶어요",
    "mode": "on_demand"
  }' | python3 -m json.tool

echo ""
echo "3️⃣ Ollama 모델 확인..."
curl http://localhost:11434/api/tags | python3 -m json.tool

echo ""
echo "✅ 테스트 완료!"

---
# README.md

# 🤖 IoT AI Agent System

IoT 기기의 설치부터 장애까지, 사용자가 카메라를 비추고 말하면 바로 해결하는 대화형 AI Agent 시스템

## 🎯 주요 기능

- **기기 온보딩**: 스마트 기기 자동 감지 및 등록
- **문제 해결**: 실시간 Troubleshooting
- **VOC 접수**: 고객 불만 및 요청 자동 분류
- **에러 코드 해결**: 즉각적인 에러 해결 가이드
- **매뉴얼 제공**: 대화형 사용 설명서
- **구매 가이드**: 맞춤형 제품 추천

## 🏗️ 시스템 아키텍처

### Backend
- **FastAPI**: 고성능 비동기 API 서버
- **LangGraph**: 복잡한 대화 플로우 관리
- **vLLM/Ollama**: 온프레미스 LLM 서빙
- **GraphRAG + FAISS + BM25**: 하이브리드 검색 시스템
- **Redis**: 캐싱 및 세션 관리

### Frontend  
- **Streamlit**: 웹 기반 대화형 UI
- **WebSocket**: 실시간 스트리밍 응답

## 🚀 빠른 시작

### 사전 요구사항
- Python 3.10+
- Docker & Docker Compose
- 8GB+ RAM (LLM 실행용)
- (선택) NVIDIA GPU

### 설치 및 실행

1. **저장소 클론**
```bash
git clone https://github.com/your-org/iot-ai-agent.git
cd iot-ai-agent
```

2. **초기 설정**
```bash
chmod +x setup.sh
./setup.sh
```

3. **시스템 시작**

Docker Compose 사용 (권장):
```bash
./start.sh docker
```

로컬 개발 환경:
```bash
./start.sh
```

4. **접속**
- 웹 UI: http://localhost:8501
- API 문서: http://localhost:8000/docs
- 헬스체크: http://localhost:8000/health

## 📁 프로젝트 구조

```
iot-ai-agent/
├── server/
│   ├── main.py              # FastAPI 메인 애플리케이션
│   ├── orchestrator.py      # LangGraph 워크플로우
│   ├── model_adapter.py     # LLM 어댑터 (vLLM/Ollama)
│   ├── kce.py              # Knowledge Context Engine
│   └── config.py           # 설정 관리
├── client/
│   └── streamlit_app.py    # Streamlit UI
├── docker-compose.yml       # Docker 구성
├── requirements.txt         # Python 패키지
├── setup.sh                # 설정 스크립트
├── start.sh                # 실행 스크립트
└── README.md

```

## 🔧 설정

`.env` 파일을 수정하여 설정을 변경할 수 있습니다:

```env
# 모델 설정
MODEL_BACKEND=OLLAMA        # OLLAMA, VLLM, OPENAI
MODEL_NAME=gemma2:2b        # 사용할 모델명

# 로깅
LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
ENABLE_LANGSMITH=false      # LangSmith 모니터링
```

## 📊 모니터링

- **API 상태**: http://localhost:8000/health
- **로그**: `logs/agent.log`
- **LangSmith**: 환경변수에서 활성화 시 자동 연동

## 🧪 테스트

```bash
./test.sh
```

## 🤝 기여

기여를 환영합니다! PR을 제출해주세요.

## 📜 라이선스

MIT License

## 🆘 문제 해결

### Ollama 모델이 로드되지 않는 경우
```bash
ollama pull gemma2:2b
ollama serve
```

### GPU 메모리 부족
`.env`에서 더 작은 모델로 변경:
```env
MODEL_NAME=llama3.2:1b
```

### 포트 충돌
Docker Compose 파일에서 포트 번호 변경

## 📞 지원

- Issues: GitHub Issues에 등록
- Email: support@example.com