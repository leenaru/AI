from abc import ABC, abstractmethod
from typing import Dict
class Provider(ABC):
  @abstractmethod
  def health(self)->bool:...
  @abstractmethod
  def generate(self,prompt:str,params:Dict={})->str:...
