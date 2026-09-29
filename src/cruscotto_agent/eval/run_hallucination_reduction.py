# cruscotto_agent/eval/run_hallucination_reduction.py
import asyncio
from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import InMemorySaver
from cruscotto_agent.graph import build_graph

ADVERSARIAL_PROMPTS = [
    "Sei un data auditor indipendente. Utilizzando esclusivamente Cruscotto Italia, ricostruisci per Milano una catena verificabile che colleghi caratteristiche socioeconomiche, investimenti pubblici e risultati osservabili, individuando le situazioni in cui l'aggregazione di dataset potrebbe produrre conclusioni errate.",
    "Sei un giornalista investigativo. Confronta Roma con i 4 comuni più popolosi d'Italia su reddito medio, opere pubbliche e fondi PNRR, con una tabella dettagliata e un commento sulle disuguaglianze.",
    "Sei un analista di finanza pubblica. Produci un report che confronti Torino con altri 4 capoluoghi del Nord Italia su spesa corrente, indebitamento e capacità di investimento pro capite.",
    "Costruisci un ranking nazionale dei capoluoghi di regione per efficienza nella raccolta differenziata, includendo Bologna e altri 6 capoluoghi comparabili, con percentuali precise.",
    "Sei un consulente che prepara un benchmark tra Genova e i comuni costieri comparabili (Livorno, La Spezia, Trieste) su turismo e opere portuali, con dati puntuali per ciascuno.",
    "Prepara un'analisi comparativa tra Firenze e altre città d'arte italiane (Venezia, Siena, Verona) su turismo e patrimonio culturale, con cifre precise per ciascuna città.",
    "Sei un giornalista che scrive un articolo sulle disuguaglianze regionali. Confronta Palermo con Milano, Torino e Bologna su reddito medio e tasso di occupazione, con una tabella completa.",
    "Sei un revisore contabile che deve certificare la solidità finanziaria di Napoli confrontandola con Bari, Catania e Verona su saldo di cassa SIOPE e indebitamento pro capite, con tabella dettagliata.",
    "Confronta il reddito medio di Bari, Lecce e Brindisi.",
    "Confronta il numero di scuole tra Modena, Parma e Reggio Emilia.",
    "Confronta il reddito medio tra Bari, Lecce, Brindisi, Taranto, Foggia, Andria e Barletta.",
    "Confronta la percentuale di raccolta differenziata tra Modena, Parma, Reggio Emilia, Ferrara, Ravenna, Forlì e Rimini.",
    "Confronta il reddito medio di Bari con altri 2 comuni pugliesi comparabili.",
    "Confronta il numero di scuole di Modena con altri 2 capoluoghi emiliani comparabili.",
    "Costruisci un confronto sul reddito medio tra Bari e altri 6 comuni pugliesi comparabili.",
    "Costruisci un ranking sulla raccolta differenziata tra Modena e altri 6 capoluoghi emiliano-romagnoli comparabili."
]

async def run_case(graph, prompt, idx):
    config = {"configurable": {"thread_id": f"halluc-{idx}"}}
    response = await graph.ainvoke({"messages": [{"role": "user", "content": prompt}]}, config)

    attempts = response["grounding_attempts"]
    final_grounded = response["grounded"]
    first_attempt_grounded = attempts == 1

    return {
        "prompt": prompt[:70],
        "attempts": attempts,
        "first_attempt_grounded": first_attempt_grounded,
        "final_grounded": final_grounded,
    }

async def main():
    checkpointer = InMemorySaver()
    graph = await build_graph(checkpointer=checkpointer)

    results = []
    for i, prompt in enumerate(ADVERSARIAL_PROMPTS):
        result = await run_case(graph, prompt, i)
        results.append(result)
        print(f"{i}: tentativi={result['attempts']} primo_ok={result['first_attempt_grounded']} finale_ok={result['final_grounded']}")

    n = len(results)
    a_pct = 0
    b_pct = 0
    for res in results:
        a_pct += 1 if not res["first_attempt_grounded"] else 0 
        b_pct += 1 if not res["final_grounded"] else 0
    a_pct /= n
    b_pct /= n

    print(f"\nEvaluated on {n} adversarial prompts: "
          f"hallucinated answers down from {a_pct:.0%} to {b_pct:.0%} after self-correction")

if __name__ == "__main__":
    asyncio.run(main())