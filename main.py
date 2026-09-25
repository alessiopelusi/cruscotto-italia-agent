import asyncio
from dotenv import load_dotenv
load_dotenv()  # deve stare qui, PRIMA degli import di cruscotto_agent

from langgraph.checkpoint.memory import InMemorySaver
from cruscotto_agent.graph import build_graph

async def main():
    checkpointer = InMemorySaver()
    graph = await build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "local-test"}}
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "Qual è la popolazione di Lecce?"}]},
        config,
    )
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())  # nota: qui è la forma corretta per lo script — non "await main()" come in Colab