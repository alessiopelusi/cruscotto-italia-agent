# cruscotto_agent/eval/inject.py
import json
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from cruscotto_agent.models import _api_key, FAST_MODEL
from cruscotto_agent.eval.schemas import InjectionCase, InjectionType

class InjectedVariants(BaseModel):
    clean_paraphrase: str = Field(description="Stessa risposta riformulata con parole diverse, zero claim nuovi o alterati")
    fabricated_comune_answer: str = Field(description="La risposta originale con aggiunta una frase su un ALTRO comune italiano mai citato nei dati, con numeri inventati")
    fabricated_comune_claims: list[str] = Field(description="La/le frasi esatte aggiunte per il caso fabricated_comune")
    wrong_value_answer: str = Field(description="La risposta originale con UN valore reale alterato a un numero diverso ma plausibile, stesso comune")
    wrong_value_claims: list[str] = Field(description="La frase esatta con il valore alterato")
    fabricated_derived_answer: str = Field(description="La risposta originale con aggiunta un'affermazione calcolata (rank, trend, rapporto) non verificabile dai dati grezzi forniti")
    fabricated_derived_claims: list[str] = Field(description="La frase esatta con l'affermazione calcolata non verificabile")

INJECTOR_SYSTEM_PROMPT = """Sei un generatore di casi di test per un sistema anti-allucinazione.
Ricevi dati grezzi JSON e una risposta corretta basata su quei dati. Il tuo compito è produrre
quattro varianti secondo istruzioni precise.

REGOLA DI STILE FONDAMENTALE: ogni claim iniettato deve essere fuso nella struttura esistente
della risposta, non aggiunto come frase separata. Se la risposta contiene una tabella o un
elenco puntato, il claim fabbricato deve essere una riga/voce in più nello STESSO formato,
senza introduzioni. Se la risposta è prosa continua, il claim va agganciato con una virgola o
una proposizione incidentale a una frase già esistente, non come frase autonoma.
VIETATO iniziare il claim iniettato con connettivi che lo segnalano come un'aggiunta separata,
come "Inoltre", "In aggiunta", "Va anche detto che", "Si segnala inoltre". Il claim deve
leggersi come se fosse sempre stato lì, indistinguibile per stile dal resto del testo.

In OGNI variante modificata, riporta in modo ESATTO (copia-incolla, non parafrasato) le frasi
che hai aggiunto o alterato, così da poterle confrontare in seguito. Le frasi non toccate della
risposta originale devono restare identiche."""

async def inject_variants(seed: dict) -> InjectedVariants:
    model = ChatGoogleGenerativeAI(model=FAST_MODEL, google_api_key=_api_key()).with_structured_output(InjectedVariants)
    user_prompt = f"""DATI GREZZI:
        {json.dumps(seed['raw_data'], ensure_ascii=False)}

        RISPOSTA ORIGINALE (corretta):
        {seed['answer_text']}"""
    return await model.ainvoke([SystemMessage(INJECTOR_SYSTEM_PROMPT), HumanMessage(user_prompt)])

def build_cases(seed: dict, variants: InjectedVariants, idx: int) -> list[InjectionCase]:
    injection_cases = []
    for injection_type in InjectionType:
        injection_cases.append(InjectionCase(
            id=f"{idx}_{injection_type.value}",
            source_prompt=seed["prompt"],
            injection_type=injection_type,
            raw_data=seed["raw_data"],
            answer_text=variants.clean_paraphrase if injection_type.value == "clean" else getattr(variants, f"{injection_type.value}_answer"),
            injected_claims=[] if injection_type.value == "clean" else getattr(variants, f"{injection_type.value}_claims"),
        ))
    return injection_cases

async def main():
    with open("eval_data/seeds.json", encoding="utf-8") as f:
        seeds = json.load(f)
    all_cases = []
    for i, seed in enumerate(seeds):
        variants = await inject_variants(seed)
        all_cases.extend(build_cases(seed, variants, i))
    with open("eval_data/injection_cases_hard_draft.json", "w", encoding="utf-8") as f:
        json.dump([c.model_dump(mode="json") for c in all_cases], f, ensure_ascii=False, indent=2)
    print(f"{len(all_cases)} casi generati in eval_data/injection_cases_draft.json")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())