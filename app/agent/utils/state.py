from typing import Annotated

from langgraph.graph import add_messages
from langgraph.graph.message import Messages
from pydantic import BaseModel


class AgentState(BaseModel):
    messages: Annotated[Messages, add_messages]
    current_tool: str | None = None
    iteration_counter: int = 0
    save_point: bool = False
