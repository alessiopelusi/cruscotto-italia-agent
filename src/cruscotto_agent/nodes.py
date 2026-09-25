# cruscotto_agent/nodes.py
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState
from cruscotto_agent.models import judge_model, classifier_model

JUDGE_SYSTEM_PROMPT = """Sei un verificatore di accuratezza fattuale. Ricevi:
    1. Dati grezzi in JSON, presi da fonti ufficiali italiane.
    2. Una risposta in linguaggio naturale generata da un assistente sulla base di quei dati.

    Verifica se OGNI affermazione fattuale o numerica della risposta è supportata dai dati.
    Sii rigoroso: se un dato non compare nei dati forniti, o li contraddice, segnalalo come
    non supportato. Non usare conoscenza tua che non sia nei dati forniti."""

CLASSIFIER_SYSTEM_PROMPT = """Sei un classificatore. Il sistema a valle risponde SOLO a domande
    su dati pubblici di comuni italiani (popolazione, redditi, opere pubbliche, scuole, ambiente,
    ecc., tramite il servizio Cruscotto Italia). Ti viene fornita l'intera conversazione fino a ora:
    classifica se l'ULTIMO messaggio dell'utente rientra in questo ambito, tenendo conto del
    contesto dei turni precedenti se necessario per capirlo."""

def extract_tool_data(messages):
    """Ritorna una lista di dict con i dati grezzi restituiti da tutti i ToolMessage."""
    raw_data = []
    for msg in messages:
        if msg.type == "tool":
          raw_data.append(json.loads(msg.content[0]["text"]))
    return raw_data

async def verify_node(state):
    """Verifica la veridicità dei dati presenti nell'ultima risposta controllando i dati grezzi restituiti da tutti i ToolMessage."""
    last_answer = state["messages"][-1].content
    if isinstance(last_answer, list):
        last_answer_text = " ".join(b.get("text", "") for b in last_answer if isinstance(b, dict))
    else:
        last_answer_text = last_answer
    raw_data = extract_tool_data(state["messages"])

    user_prompt = f"""DATI GREZZI:
        {json.dumps(raw_data, ensure_ascii=False)}

        RISPOSTA DA VERIFICARE:
        {last_answer_text}"""
        
    verdict = await judge_model.ainvoke([SystemMessage(JUDGE_SYSTEM_PROMPT), HumanMessage(user_prompt)])

    if verdict.grounded:
        print("✅ Verificato dal giudice LLM")
    else:
        print(f"⚠️ Claim non supportati secondo il giudice: {verdict.unsupported_claims}")
       
    return {}

async def classify_intent(state):
    result = await classifier_model.ainvoke([SystemMessage(CLASSIFIER_SYSTEM_PROMPT)] + state["messages"])
    return {"in_scope": result.in_scope}

def route_by_scope(state):
    if state["in_scope"]:
        return "agent"
    return "out_of_scope"

async def out_of_scope_node(state):
    msg = AIMessage(content="Mi occupo solo di dati pubblici sui comuni italiani (Cruscotto Italia) — popolazione, redditi, opere pubbliche e simili. Posso aiutarti con una domanda su questo?")
    return {"messages": [msg]}

def make_call_model_node(model_with_tools):
    async def call_model(state: MessagesState):
      response = await model_with_tools.ainvoke(state["messages"])
      return {"messages": [response]}
    return call_model