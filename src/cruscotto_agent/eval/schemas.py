# cruscotto_agent/eval/schemas.py
from pydantic import BaseModel, Field
from typing import Literal

from enum import Enum

class InjectionType(Enum):
    FABRICATED_COMUNE = "fabricated_comune"
    WRONG_VALUE = "wrong_value"
    FABRICATED_DERIVED = "fabricated_derived"
    CLEAN = "clean"
    
class InjectionCase(BaseModel):
    id: str
    source_prompt: str
    injection_type: InjectionType
    raw_data: list[dict]
    answer_text: str
    injected_claims: list[str] = Field(default_factory=list)