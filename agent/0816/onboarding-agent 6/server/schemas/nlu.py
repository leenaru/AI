from typing import Dict, Optional, List
from pydantic import BaseModel, Field

class SafetyInfo(BaseModel):
    pii: List[str] = []
    disallowed: bool = False

class NLU(BaseModel):
    label: str = Field(pattern=r"^(faq|troubleshooting|device_registration|smalltalk|other)$")
    score: float = Field(ge=0, le=1)
    slots: Dict[str, str] = {}
    normalized_text: Optional[str] = None
    need_image: Optional[bool] = None
    safety: Optional[SafetyInfo] = None
