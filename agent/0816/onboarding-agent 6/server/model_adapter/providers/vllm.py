import httpx
from server.model_adapter.base import Provider
class VLLMProvider(Provider):
  def __init__(self, base_url: str = "http://vllm:8000/v1", model: str = "qwen72b"):
    self.base_url, self.model = base_url, model
  def health(self)->bool:
    try:
      r=httpx.get(f"{self.base_url}/models",timeout=2); return r.status_code==200
    except Exception: return False
  def generate(self, prompt: str, params: dict = {})->str:
    payload={"model":self.model,"messages":[{"role":"user","content":prompt}],"max_tokens":512}
    payload.update(params)
    r=httpx.post(f"{self.base_url}/chat/completions",json=payload,timeout=60); r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
