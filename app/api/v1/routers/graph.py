import logging
import uuid
from typing import Final

from fastapi import APIRouter
from fastapi.exceptions import HTTPException

from app.agent.graph import build_graph, get_checkpointer
from app.agent.utils.llm import HumanMessage, SystemMessage

from ...models.graph import (
    RequestConvGraph,
    RequestResetMessagesGraph,
    RequestStartGraph,
    RequestStateGraph,
    ResponseConvGraph,
    ResponseStartGraph,
)

router = APIRouter(prefix="/graph", tags=["Graph"])
logger = logging.getLogger("app.api.v1.graph")

SYSTEM_PROMPT: Final = "Ты умеешь ВСЕ!"


def check_is_first_conv(graph_state):
    messages = graph_state.values.get("messages")
    if not isinstance(messages[0], SystemMessage):
        raise HTTPException(
            status_code=400, detail="You must first request to '/graph/start' !"
        )


async def worker_graph(uid_conversation: str, messages: list):
    config = {"configurable": {"thread_id": uid_conversation}, "recursion_limit": 10}
    payload = {"messages": messages}

    graph = build_graph()
    await graph.aupdate_state(config, values=payload)
    graph_state = await graph.aget_state(config)
    logger.warning(f"{graph_state=}")

    check_is_first_conv(graph_state)

    try:
        agent_state = await graph.ainvoke({"messages": []}, config)
    except Exception as e:
        logger.error(f"Error by graph.invoke: {e}")
        raise HTTPException(status_code=503, detail=str(e))

    return agent_state


@router.post("/start")
async def start_graph(
    request: RequestStartGraph, system_prompt: str | None = None, debug: bool = False
):
    if not system_prompt:
        system_prompt = SYSTEM_PROMPT

    # TODO: change uid_conversation
    uid_conversation = str(uuid.uuid4())
    messages = [SystemMessage(system_prompt), HumanMessage(content=request.question)]
    agent_state = await worker_graph(uid_conversation, messages)

    messages = agent_state.get("messages")
    if not messages:
        raise HTTPException(status_code=404, detail="llm answer is not found")
    return ResponseStartGraph(
        uid_conversation=uid_conversation,
        agent_state=(agent_state if debug else None),
        ai_answer=messages[-1].content,
    )


@router.post("/conv")
async def conv_graph(request: RequestConvGraph, debug: bool = False):
    messages = [HumanMessage(content=request.question)]
    agent_state = await worker_graph(request.uid_conversation, messages)

    messages = agent_state.get("messages")
    if not messages:
        raise HTTPException(status_code=404, detail="llm answer is not found")

    return ResponseConvGraph(
        uid_conversation=request.uid_conversation,
        agent_state=(agent_state if debug else None),
        ai_answer=messages[-1].content,
    )


@router.delete("/reset", status_code=204)
async def reset_messages_graph(request: RequestResetMessagesGraph):
    checkpointer = get_checkpointer()
    try:
        await checkpointer.adelete_thread(request.uid_conversation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/state")
async def get_state_graph(
    request: RequestStateGraph,
):
    config = {
        "configurable": {"thread_id": request.uid_conversation},
        "recursion_limit": 10,
    }
    checkpointer = get_checkpointer()
    return await checkpointer.aget(config)
