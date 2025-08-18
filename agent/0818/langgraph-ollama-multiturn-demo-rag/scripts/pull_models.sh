#!/usr/bin/env bash
set -euo pipefail

# 이 스크립트는 호스트에서 실행 가능하며, docker compose 사용 시
# 컨테이너 내부에서는 'app' 서비스 시작 전 Ollama pull을 수행하려면 entrypoint 스크립트로 변경하세요.

MODELS=(
  "llama3:8b-instruct"
  "qwen2.5:3b-instruct"
  "qwen3:8b-instruct"   # 가용하지 않으면 무시될 수 있음
  "qwen2.5:7b-instruct"
  "qwen2.5:14b-instruct"
)

for m in "${MODELS[@]}"; do
  echo "Pulling $m ..."
  ollama pull "$m" || echo "Skip: $m not available"
done
