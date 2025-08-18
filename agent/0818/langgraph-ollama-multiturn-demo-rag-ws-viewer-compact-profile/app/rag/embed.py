from __future__ import annotations
import os
from langchain_community.embeddings import OllamaEmbeddings
from app.utils.model_tags import OLLAMA_HOST

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

def get_embedder():
    # OllamaEmbeddings uses OLLAMA_HOST env var implicitly if set
    return OllamaEmbeddings(model=EMBED_MODEL)
