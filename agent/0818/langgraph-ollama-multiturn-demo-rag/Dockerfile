# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential curl &&     rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# 기본 포트
EXPOSE 8000

# 런타임 환경변수 (docker-compose에서 덮어씀)
ENV OLLAMA_HOST=http://ollama:11434
ENV ROUTER_MODEL=qwen2.5:3b-instruct
ENV GENERATOR_MODEL=llama3:8b-instruct
ENV DEFAULT_MODEL=llama3:8b-instruct

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
