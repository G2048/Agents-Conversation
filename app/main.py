import asyncio

from agent.graph import CompiledStateGraph, build_graph
from agent.utils.llm import AIMessage, HumanMessage, start_context


def prepare_graph() -> tuple[CompiledStateGraph, dict]:
    id_conversation = 0
    config = {"configurable": {"thread_id": id_conversation}, "recursion_limit": 10}
    graph = build_graph()
    return graph, config


async def amain():
    list_questions = ("Привет!", "Сложи 2 + 2", "Теперь добавь еще +1")
    graph, config = prepare_graph()
    start_messages = start_context()

    start_payload = {"messages": [start_messages]}
    empty_payload = {"messages": []}

    graph.update_state(config, start_payload)
    messages = await graph.ainvoke(start_payload, config=config)
    for question in list_questions:
        print(f"Current question: {question}")
        payload = {"messages": [HumanMessage(content=question)]}
        messages = await graph.ainvoke(payload, config=config)

        current_state = graph.get_state(config)
        current_chat_state = current_state.values
        for message in current_chat_state.get("messages", []):
            if isinstance(message, HumanMessage):
                print(f"Вы: {message.content}")
            elif isinstance(message, AIMessage):
                print(f"AI: {message.content}")
        print()


if __name__ == "__main__":
    asyncio.run(amain())
