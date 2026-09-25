# cruscotto_agent/schemas.py
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

class IntentClassification(BaseModel):
    in_scope: bool = Field(description="True se la domanda riguarda dati pubblici di comuni italiani (popolazione, redditi, opere pubbliche, scuole, ambiente, ecc.); False se è fuori tema")

class GroundingVerdict(BaseModel):
    grounded: bool = Field(description="True se ogni affermazione fattuale/numerica della risposta è supportata dai dati grezzi forniti")
    unsupported_claims: list[str] = Field(default_factory=list, description="Affermazioni non supportate o in contraddizione con i dati; lista vuota se grounded=True")

class AgentState(MessagesState):
    in_scope: bool
    grounded: bool
    grounding_attempts: int