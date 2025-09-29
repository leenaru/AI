# server/model_adapter.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncGenerator
import aiohttp
import asyncio
import json
from enum import Enum

class ModelBackend(Enum):
    VLLM = "vllm"
    OLLAMA = "ollama"
    OPENAI = "openai"  # For testing

class BaseModelAdapter(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator:
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    def health_check(self) -> Dict:
        pass

class VLLMAdapter(BaseModelAdapter):
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name
        self.headers = {"Content-Type": "application/json"}
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """vLLM 생성"""
        url = f"{self.base_url}/v1/completions"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["text"]
                else:
                    raise Exception(f"vLLM error: {await response.text()}")
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator:
        """vLLM 스트리밍 생성"""
        url = f"{self.base_url}/v1/completions"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self.headers) as response:
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8').strip().replace('data: ', ''))
                            if 'choices' in data:
                                yield data['choices'][0]['text']
                        except:
                            continue
    
    async def embed(self, text: str) -> List[float]:
        """텍스트 임베딩"""
        url = f"{self.base_url}/v1/embeddings"
        payload = {
            "model": self.model_name,
            "input": text
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["data"][0]["embedding"]
                else:
                    raise Exception(f"vLLM embedding error: {await response.text()}")
    
    def health_check(self) -> Dict:
        return {"backend": "vLLM", "status": "operational", "model": self.model_name}

class OllamaAdapter(BaseModelAdapter):
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Ollama 생성"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.95),
                "num_predict": kwargs.get("max_tokens", 1024)
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["response"]
                else:
                    raise Exception(f"Ollama error: {await response.text()}")
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator:
        """Ollama 스트리밍 생성"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 1024)
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                yield data['response']
                        except:
                            continue
    
    async def embed(self, text: str) -> List[float]:
        """텍스트 임베딩"""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["embedding"]
                else:
                    raise Exception(f"Ollama embedding error: {await response.text()}")
    
    def health_check(self) -> Dict:
        return {"backend": "Ollama", "status": "operational", "model": self.model_name}

class OpenAIAdapter(BaseModelAdapter):
    """OpenAI API adapter for testing"""
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate(self, prompt: str, **kwargs) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"OpenAI error: {await response.text()}")
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self.headers) as response:
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str != '[DONE]':
                                try:
                                    data = json.loads(data_str)
                                    if 'choices' in data and data['choices'][0]['delta'].get('content'):
                                        yield data['choices'][0]['delta']['content']
                                except:
                                    continue
    
    async def embed(self, text: str) -> List[float]:
        url = f"{self.base_url}/embeddings"
        payload = {
            "model": "text-embedding-ada-002",
            "input": text
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["data"][0]["embedding"]
                else:
                    raise Exception(f"OpenAI embedding error: {await response.text()}")
    
    def health_check(self) -> Dict:
        return {"backend": "OpenAI", "status": "operational", "model": self.model_name}

class ModelAdapter:
    """통합 Model Adapter"""
    def __init__(self, settings):
        self.settings = settings
        self.backend = self._initialize_backend()
        self.fallback_backend = self._initialize_fallback_backend()
    
    def _initialize_backend(self) -> BaseModelAdapter:
        """주 백엔드 초기화"""
        backend_type = self.settings.model_backend
        
        if backend_type == ModelBackend.VLLM:
            return VLLMAdapter(
                base_url=self.settings.vllm_base_url,
                model_name=self.settings.model_name
            )
        elif backend_type == ModelBackend.OLLAMA:
            return OllamaAdapter(
                base_url=self.settings.ollama_base_url,
                model_name=self.settings.model_name
            )
        elif backend_type == ModelBackend.OPENAI:
            return OpenAIAdapter(
                api_key=self.settings.openai_api_key,
                model_name=self.settings.model_name
            )
        else:
            raise ValueError(f"Unknown backend: {backend_type}")
    
    def _initialize_fallback_backend(self) -> Optional[BaseModelAdapter]:
        """폴백 백엔드 초기화"""
        if self.settings.fallback_backend:
            if self.settings.fallback_backend == ModelBackend.OLLAMA:
                return OllamaAdapter(
                    base_url=self.settings.ollama_base_url,
                    model_name=self.settings.fallback_model_name
                )
        return None
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """프롬프트 생성 with fallback"""
        try:
            # 프롬프트 템플릿 적용
            formatted_prompt = self._format_prompt(prompt)
            return await self.backend.generate(formatted_prompt, **kwargs)
        except Exception as e:
            print(f"Primary backend failed: {e}")
            if self.fallback_backend:
                print("Trying fallback backend...")
                return await self.fallback_backend.generate(prompt, **kwargs)
            raise e
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator:
        """스트리밍 생성"""
        try:
            formatted_prompt = self._format_prompt(prompt)
            async for chunk in self.backend.generate_stream(formatted_prompt, **kwargs):
                yield chunk
        except Exception as e:
            print(f"Streaming failed: {e}")
            if self.fallback_backend:
                async for chunk in self.fallback_backend.generate_stream(prompt, **kwargs):
                    yield chunk
            else:
                raise e
    
    async def embed(self, text: str) -> List[float]:
        """텍스트 임베딩"""
        try:
            return await self.backend.embed(text)
        except Exception as e:
            if self.fallback_backend:
                return await self.fallback_backend.embed(text)
            raise e
    
    def _format_prompt(self, prompt: str) -> str:
        """프롬프트 포맷팅"""
        # 모델별 프롬프트 템플릿 적용
        if "gemma" in self.settings.model_name.lower():
            return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        elif "llama" in self.settings.model_name.lower():
            return f"[INST] {prompt} [/INST]"
        else:
            return prompt
    
    def health_check(self) -> Dict:
        """헬스 체크"""
        return {
            "primary": self.backend.health_check(),
            "fallback": self.fallback_backend.health_check() if self.fallback_backend else None
        }