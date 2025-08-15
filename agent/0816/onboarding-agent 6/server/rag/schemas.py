from typing import Dict, Any
from pydantic import BaseModel
class Passage(BaseModel):
  id:str; text:str; meta:Dict[str,Any]={}; score:float=0.0
