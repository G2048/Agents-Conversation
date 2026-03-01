import asyncio

from agent.graph import CompiledStateGraph, build_graph
from agent.utils.llm import AIMessage, HumanMessage, start_context


def prepare_graph() -> tuple[CompiledStateGraph, dict]:
    id_conversation = 0
    config = {"configurable": {"thread_id": id_conversation}, "recursion_limit": 10}
    graph = build_graph()
    return graph, config


def print_dialog(chat_state):
    for message in chat_state.get("messages", []):
        if isinstance(message, HumanMessage):
            print(f"Вы: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"AI: {message.content[:100]}")


async def amain():
    list_questions = ("Привет!", "Сложи 2 + 2", "Теперь добавь еще +1", "Напиши только слово DIXI")
    graph, config = prepare_graph()
    start_messages = start_context()

    # Первым сообщнением обязательно нужно отправить SystemPrompt + HumanMessage !
    # Warning: если не обозначить прямо сообщение как HumanMessage - будет задвоение сообщeния!
    iteration_counter = 0
    for counter, question in enumerate(list_questions):
        print(f"Current question: {question}")
        if counter == 0:
            payload = {"messages": [start_messages, HumanMessage(content=question)]}
        else:
            payload = {"messages": [HumanMessage(content=question)]}

        current_chat_state = await graph.ainvoke(payload, config=config)
        iteration_counter += 1
        print_dialog(current_chat_state)
        print("--end--\n\n")
    print(f"{iteration_counter=}")


if __name__ == "__main__":
    asyncio.run(amain())
