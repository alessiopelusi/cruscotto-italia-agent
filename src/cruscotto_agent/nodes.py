# cruscotto_agent/nodes.py
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import END, MessagesState
from cruscotto_agent.models import judge_model, classifier_model

MAX_GROUNDING_ATTEMPTS = 2  

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
    last_answer_text = extract_answer_text(state["messages"][-1].content)
    raw_data = extract_tool_data(state["messages"])

    user_prompt = f"""DATI GREZZI:
        {json.dumps(raw_data, ensure_ascii=False)}

        RISPOSTA DA VERIFICARE:
        {last_answer_text}"""
        
    verdict = await judge_model.ainvoke([SystemMessage(JUDGE_SYSTEM_PROMPT), HumanMessage(user_prompt)])

    attempts = state.get("grounding_attempts", 0) + 1

    if verdict.grounded:
        print("Verificato dal giudice LLM")
        return {"grounded": True, "grounding_attempts": attempts}
    else:
        print(f"Claim non supportati secondo il giudice: {verdict.unsupported_claims}")
        if attempts >= MAX_GROUNDING_ATTEMPTS:
            alert_message = AIMessage(f"Nota: alcune affermazioni non sono state verificabili rispetto ai dati disponibili: {verdict.unsupported_claims}")
            return {"messages": [alert_message], "grounded": False, "grounding_attempts": attempts}
    correction_message  = HumanMessage(f"""La risposta precedente conteneva affermazioni non supportate dai dati:
        {verdict.unsupported_claims}

        Riformula la risposta. Se hai già a disposizione i dati corretti, usali. Se invece la
        risposta richiede dati che non hai ancora recuperato (es. un comune non ancora
        interrogato), chiama i tool necessari per recuperarli prima di rispondere di nuovo.
        Se un dato non è comunque ottenibile con gli strumenti disponibili, dichiaralo
        esplicitamente invece di inventarlo.""")
    return {"messages": [correction_message], "grounded": False, "grounding_attempts": attempts}

async def classify_intent(state):
    result = await classifier_model.ainvoke([SystemMessage(CLASSIFIER_SYSTEM_PROMPT)] + state["messages"])
    return {"in_scope": result.in_scope, "grounding_attempts": 0, "grounded": False}

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


def route_after_verify(state):
    if state["grounded"] or state["grounding_attempts"] >= MAX_GROUNDING_ATTEMPTS:
        return END
    return "agent"

def extract_answer_text(content):
    """Estrae il testo da un content di AIMessage, che può essere str o list[dict]."""
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content