import asyncio

from .llm import get_llm
from .state import AgentState, ConversationState


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
    state.eof = True
    return state


# TODO: заменить на Command(update={"foo": "baz"}, goto="my_other_node")
def check_eof_node(state: AgentState) -> str:
    print(f"{state.iteration_counter=}")
    if ConversationState.EOF in ((state and state.messages[-1].content) or ()):
        return "save_state"
    return "end"


def end_node(state: AgentState):
    return state
