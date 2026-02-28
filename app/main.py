import asyncio

from agent.graph import CompiledStateGraph, build_graph
from agent.utils.llm import AIMessage, HumanMessage, start_context


def prepare_graph() -> tuple[CompiledStateGraph, dict]:
    id_conversation = 0
    config = {"configurable": {"thread_id": id_conversation}, "recursion_limit": 10}
    graph = build_graph()
    return graph, config


async def amain():
    list_questions = ("Привет!", "Сложи 2 + 2", "Теперь добавь еще +1", "Напиши только слово DIXI")
    graph, config = prepare_graph()
    start_messages = start_context()

    for counter, question in enumerate(list_questions):
        print(f"Current question: {question}")
        payload = {"messages": [HumanMessage(content=question)]}
        if counter == 0:
            payload = {"messages": [start_messages, HumanMessage(content=question)]}

        async for current_node in graph.astream(payload, config=config):
            print(f"{current_node=}\n")
            current_chat_state = current_node.get("save_state", {})
            # current_state = graph.get_state(config)
            # current_chat_state = current_state.values
            for message in current_chat_state.get("messages", []):
                if isinstance(message, HumanMessage):
                    print(f"Вы: {message.content}")
                elif isinstance(message, AIMessage):
                    print(f"AI: {message.content}")
        print("--end--\n\n")


if __name__ == "__main__":
    asyncio.run(amain())
