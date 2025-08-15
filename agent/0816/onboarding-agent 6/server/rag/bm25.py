from rank_bm25 import BM25Okapi
from typing import List
import re
def tokenize(text: str)->List[str]: return re.findall(r"\w+", text.lower())
class BM25:
  def __init__(self, corpus: List[str]):
    self.tokens=[tokenize(t) for t in corpus]
    self.model=BM25Okapi(self.tokens)
  def search(self, q: str, k: int = 10):
    scores=self.model.get_scores(tokenize(q))
    idx=sorted(range(len(scores)), key=lambda i:scores[i], reverse=True)[:k]
    return idx,[scores[i] for i in idx]
