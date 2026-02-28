import asyncio

from .llm import get_llm
from .state import AgentState


def should_continue(state):
    if state["retry_count"] > 3:
        return "end"
    elif state["current_tool"] == "search":
        return "process_search"
    else:
        return "call_llm"


def human_node(state: AgentState):
    return state


async def agent_node(state: AgentState):
    count = state.iteration_counter + 1
    llm = get_llm()
    request = await llm.ainvoke(state.messages)  # ty: ignore
    return {"messages": [request], "iteration_counter": count}
    # return state


async def save_state_node(state: AgentState):
    print("SAVE STATE IN DATABASE")
    await asyncio.sleep(1)
    print("DATA SAVED!")
    return state


# TODO: заменить на Command(update={"foo": "baz"}, goto="my_other_node")
def check_eof_node(state: AgentState) -> str:
    print(f"{state.iteration_counter=}")
    if state.iteration_counter >= 4:
        return "save_state"
    return "human"
