# server/config.py
from pydantic import BaseSettings
from typing import Optional
from model_adapter import ModelBackend

class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # Server Settings
    app_name: str = "IoT AI Agent"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model Settings
    model_backend: ModelBackend = ModelBackend.OLLAMA  # 개발: OLLAMA, 운영: VLLM
    model_name: str = "gemma2:2b"  # Ollama: gemma2:2b, vLLM: google/gemma-2b
    embedding_dim: int = 768
    
    # vLLM Settings
    vllm_base_url: str = "http://localhost:8001"
    vllm_model_path: str = "/models/gemma-2b"
    
    # Ollama Settings
    ollama_base_url: str = "http://localhost:11434"
    
    # OpenAI Settings (for testing)
    openai_api_key: Optional[str] = None
    
    # Fallback Settings
    fallback_backend: Optional[ModelBackend] = ModelBackend.OLLAMA
    fallback_model_name: str = "llama3.2:1b"
    
    # RAG Settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_retrieval: int = 5
    
    # Graph Settings
    max_graph_depth: int = 3
    graph_similarity_threshold: float = 0.7
    
    # Cache Settings
    enable_cache: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Logging Settings
    log_level: str = "INFO"
    log_file: str = "logs/agent.log"
    enable_langsmith: bool = False
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "iot-agent"
    
    # Security Settings
    enable_pii_filtering: bool = True
    enable_content_policy: bool = True
    max_message_length: int = 4000
    
    # Session Settings
    session_timeout: int = 1800  # 30 minutes
    max_session_history: int = 100
    
    # Database Settings (future)
    database_url: Optional[str] = "sqlite:///./agent.db"
    redis_url: Optional[str] = "redis://localhost:6379"
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# 환경별 설정
class DevelopmentSettings(Settings):
    """개발 환경 설정"""
    debug: bool = True
    model_backend: ModelBackend = ModelBackend.OLLAMA
    log_level: str = "DEBUG"

class ProductionSettings(Settings):
    """운영 환경 설정"""
    debug: bool = False
    model_backend: ModelBackend = ModelBackend.VLLM
    log_level: str = "INFO"
    enable_langsmith: bool = True
    enable_cache: bool = True

class TestSettings(Settings):
    """테스트 환경 설정"""
    debug: bool = True
    model_backend: ModelBackend = ModelBackend.OPENAI
    log_level: str = "DEBUG"
    
def get_settings(env: str = "development") -> Settings:
    """환경별 설정 반환"""
    if env == "production":
        return ProductionSettings()
    elif env == "test":
        return TestSettings()
    else:
        return DevelopmentSettings()