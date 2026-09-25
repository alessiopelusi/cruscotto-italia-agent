# cruscotto_agent/mcp_setup.py
from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_URL = "https://cruscotto-italia-mcp.agid.workers.dev/mcp"
ALLOWED_TOOLS = ["search_comune", "comune_kpi"]

async def get_mvp_tools():
    client = MultiServerMCPClient({
        "cruscotto_italia": {"url": MCP_URL, "transport": "streamable_http"},
    })
    tools = await client.get_tools()
    return [t for t in tools if t.name in ALLOWED_TOOLS]