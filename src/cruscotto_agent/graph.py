# cruscotto_agent/graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from cruscotto_agent.schemas import AgentState
from cruscotto_agent.mcp_setup import get_mvp_tools
from cruscotto_agent.models import make_answering_model
from cruscotto_agent.nodes import (
    classify_intent, route_by_scope, out_of_scope_node,
    verify_node, make_call_model_node, route_after_verify
)

async def build_graph(checkpointer=None):
    tools = await get_mvp_tools()
    model_with_tools = make_answering_model().bind_tools(tools)
    
    graph = StateGraph(AgentState)
    graph.add_node("agent", make_call_model_node(model_with_tools))
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("verify", verify_node)
    graph.add_node("classify", classify_intent)
    graph.add_node("out_of_scope", out_of_scope_node)
    
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_by_scope)
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "verify"})
    graph.add_conditional_edges("verify", route_after_verify)
    graph.add_edge("tools", "agent")
    graph.add_edge("out_of_scope", END)

    
    return graph.compile(checkpointer=checkpointer)
    
     
    