import asyncio

from agent.graph import CompiledStateGraph, build_graph
from agent.utils.llm import AIMessage, HumanMessage, start_context


def prepare_graph() -> tuple[CompiledStateGraph, dict]:
    id_conversation = 0
    config = {"configurable": {"thread_id": id_conversation}, "recursion_limit": 10}
    graph = build_graph()
    return graph, config


def print_dialog(chat_state):
    print("\n\n--start--")
    for message in chat_state.get("messages", []):
        if isinstance(message, HumanMessage):
            print(f"Вы: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"AI: {message.content[:100]}")
        print("--end--\n\n")


async def amain():
    list_questions = ("Привет!", "Сложи 2 + 2", "Теперь добавь еще +1", "Напиши только слово DIXI")
    graph, config = prepare_graph()
    start_messages = start_context()

    iteration_counter = 0
    for counter, question in enumerate(list_questions):
        print(f"Current question: {question}")
        payload = {"messages": [HumanMessage(content=question)]}
        if counter == 0:
            payload = {"messages": [start_messages, HumanMessage(content=question)]}

        async for current_node in graph.astream(payload, config=config):
            # print(f"{current_node=}\n")
            current_chat_state = current_node.get("save_state", {})
            current_counter = current_chat_state.get("iteration_counter")
            if current_counter:
                iteration_counter += current_counter
            # current_state = graph.get_state(config)
            # current_chat_state = current_state.values
            print_dialog(current_chat_state)
    print(f"{iteration_counter=}")


if __name__ == "__main__":
    asyncio.run(amain())
