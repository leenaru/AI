import httpx
from server.model_adapter.base import Provider

class OllamaProvider(Provider):
    def __init__(self, base_url: str = "http://ollama:11434", model: str = "llama3"):
        self.base_url, self.model = base_url, model

    def health(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, params: dict = {}) -> str:
        payload = {"model": self.model, "prompt": prompt, **params}
        r = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
        r.raise_for_status()
        return r.text
