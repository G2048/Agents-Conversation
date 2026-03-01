from typing import Any

from pydantic import BaseModel

type UidConversation = str


class RequestStartGraph(BaseModel):
    question: str


class ResponseStartGraph(BaseModel):
    uid_conversation: UidConversation
    ai_answer: str
    agent_state: Any | None = None


class RequestConvGraph(RequestStartGraph):
    uid_conversation: UidConversation


class ResponseConvGraph(ResponseStartGraph):
    pass


class RequestResetMessagesGraph(BaseModel):
    uid_conversation: UidConversation


class RequestStateGraph(BaseModel):
    uid_conversation: UidConversation
