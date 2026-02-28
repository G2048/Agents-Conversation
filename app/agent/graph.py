from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from .utils.nodes import agent_node, check_eof_node, human_node, save_state_node
from .utils.state import AgentState

checkpointer = InMemorySaver()


def build_graph() -> CompiledStateGraph:
    return (
        StateGraph(AgentState)
        .add_node("human", human_node)
        .add_node("agent", agent_node)
        .add_node("save_state", save_state_node)

        .set_entry_point("human")
        # .set_entry_point("agent")
        .add_edge("human", "agent")
        .add_conditional_edges("agent", check_eof_node)
        .set_finish_point("save_state")
        # .compile(checkpointer=checkpointer, interrupt_before=["human"])
        # .compile(checkpointer=checkpointer, interrupt_before=["agent"])
        .compile(checkpointer=checkpointer)
    )
