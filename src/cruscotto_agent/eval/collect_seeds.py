# cruscotto_agent/eval/collect_seeds.py
import asyncio
import json
from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import InMemorySaver
from cruscotto_agent.graph import build_graph
from cruscotto_agent.nodes import extract_tool_data, extract_answer_text

SEED_PROMPTS = [
    "Qual è la popolazione di Bari?",
    "Confronta il reddito medio di Torino e Genova",
    "Quante scuole ci sono a Palermo?",
    "Qual è il tasso di disoccupazione di Catania?",
    "Confronta la percentuale di raccolta differenziata tra Firenze e Bologna",
    "Quanti punti di ricarica elettrica ci sono a Verona?",
    "Qual è l'indice di vecchiaia di Venezia?",
    "Confronta il numero di opere pubbliche tra Padova e Trieste",
]

async def collect_seeds():
    checkpointer = InMemorySaver()
    graph = await build_graph(checkpointer=checkpointer)
    seeds = []
    for i, prompt in enumerate(SEED_PROMPTS):
        config = {"configurable": {"thread_id": f"seed-{i}"}}
        response = await graph.ainvoke({"messages": [{"role": "user", "content": prompt}]}, config)
        if not response.get("grounded", False):
            print(f"Scartato (non grounded al primo giro): {prompt}")
            continue
        seeds.append({
            "prompt": prompt,
            "raw_data": extract_tool_data(response["messages"]),
            "answer_text": extract_answer_text(response["messages"][-1].content),
        })
        print(f"OK: {prompt}")
    with open("eval_data/seeds.json", "w", encoding="utf-8") as f:
        json.dump(seeds, f, ensure_ascii=False, indent=2)
    print(f"\n{len(seeds)} seed puliti salvati.")

if __name__ == "__main__":
    asyncio.run(collect_seeds())