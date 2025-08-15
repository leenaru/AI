import yaml
from typing import Dict
from server.model_adapter.providers.ollama import OllamaProvider
from server.model_adapter.providers.vllm import VLLMProvider

class ModelRouter:
    def __init__(self, providers: Dict[str, object], prefer: list):
        self.providers = providers
        self.prefer = prefer

    @classmethod
    def load_from_yaml(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        providers = {
            "ollama": OllamaProvider(model=cfg["ollama"]["model"], base_url=cfg["ollama"]["base_url"]),
            "vllm": VLLMProvider(model=cfg["vllm"]["model"], base_url=cfg["vllm"]["base_url"]),
        }
        return cls(providers, cfg.get("prefer", ["ollama", "vllm"]))

    def generate(self, prompt: str) -> str:
        for key in self.prefer:
            prv = self.providers[key]
            if prv.health():
                try:
                    return prv.generate(prompt)
                except Exception:
                    continue
        # last try
        return list(self.providers.values())[-1].generate(prompt)
