from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama

__all__ = ("AIMessage", "AIMessageChunk", "HumanMessage", "SystemMessage", "get_llm", "start_context")


@lru_cache(maxsize=16)
def get_llm() -> BaseChatModel:
    # model = "llama3.1"
    model = "deepseek-r1"
    llm = ChatOllama(
        model=model,
        temperature=0,
    )
    return llm


def start_context():
    return SystemMessage(content="Ты умеешь все.")


if __name__ == "__main__":
    llm = get_llm()
    answer = llm.invoke([HumanMessage(content="Привет!")])
    print(f"{answer=}")
