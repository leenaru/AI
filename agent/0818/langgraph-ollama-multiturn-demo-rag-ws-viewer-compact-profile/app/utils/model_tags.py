import os

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "qwen2.5:3b-instruct")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "llama3:8b-instruct")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3:8b-instruct")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
