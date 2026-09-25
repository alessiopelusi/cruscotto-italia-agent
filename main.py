import asyncio
from dotenv import load_dotenv
load_dotenv()  # deve stare qui, PRIMA degli import di cruscotto_agent

from langgraph.checkpoint.memory import InMemorySaver
from cruscotto_agent.graph import build_graph

async def main():
    checkpointer = InMemorySaver()
    graph = await build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "local-test"}}
    
    # Response 1
    response1 = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "Sei un data auditor indipendente. Utilizzando esclusivamente Cruscotto Italia, ricostruisci per Milano una catena verificabile che colleghi caratteristiche socioeconomiche → bisogni territoriali → investimenti pubblici → contratti → opere/progetti → flussi finanziari → risultati osservabili, e dimostra per ogni passaggio se il collegamento è direttamente osservato nei dati, indirettamente derivato oppure non dimostrabile. Individua inoltre tutte le situazioni in cui l'aggregazione di dataset apparentemente compatibili potrebbe produrre una conclusione statisticamente o semanticamente errata."}]},
        config,
    )
    for msg in response1["messages"]:
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            print([tc["name"] + " " + str(tc["args"]) for tc in msg.tool_calls])
    print(response1["messages"][-1].content)

    # Response 2
    response2 = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "Non voglio una metodologia né una spiegazione generale. Voglio le evidenze. Esegui effettivamente l'audit su Milano. Per ogni affermazione fornisci dataset, campo, periodo, valore, identificativo utilizzato per il join e tool MCP interrogato. Ricostruisci almeno 5 catene CUP → CIG → contratto → opera → pagamento e indica per ciascuna quali collegamenti sono verificati e quali no. Poi esegui un confronto con 5 comuni comparabili e un'analisi sub-comunale sulle sezioni censuarie. Se un dato non è disponibile tramite MCP, dichiaralo esplicitamente invece di inferirlo."}]},
        config
    )
    for msg in response2["messages"]:
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            print([tc["name"] + " " + str(tc["args"]) for tc in msg.tool_calls])
    for mess in response2["messages"]:
        if mess.type != "tool":
            print(mess)

if __name__ == "__main__":
    asyncio.run(main())  # nota: qui è la forma corretta per lo script — non "await main()" come in Colab