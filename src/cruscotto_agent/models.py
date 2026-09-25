# cruscotto_agent/models.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from cruscotto_agent.schemas import IntentClassification, GroundingVerdict

FAST_MODEL = "gemini-3.1-flash-lite"

def _api_key():
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY non trovata: controlla il file .env")
    return key

def make_answering_model():
    return ChatGoogleGenerativeAI(model=FAST_MODEL, google_api_key=_api_key())

judge_model = ChatGoogleGenerativeAI(model=FAST_MODEL, google_api_key=_api_key()).with_structured_output(GroundingVerdict)
classifier_model = ChatGoogleGenerativeAI(model=FAST_MODEL, google_api_key=_api_key()).with_structured_output(IntentClassification)