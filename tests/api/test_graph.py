from fastapi.testclient import TestClient

from app.api.server import app

HOST = "http://localhost:8000"
API = "/api/v1"


list_questions = ("Привет!", "Сложи 2 + 2", "Теперь добавь еще +1", "Напиши только слово DIXI")
EXPECTED_EOF = "DIXI"


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_graph():
    base_url = HOST + API + "/graph"
    debug = True
    query = {"debug": debug}
    with TestClient(app) as client:
        question = list_questions[0]
        payload = {"question": question}
        response = client.post(base_url + "/start", json=payload, params=query)
        assert response.status_code == 200

        response_json = response.json()
        uid_conversation = response_json.get("uid_conversation")
        assert uid_conversation

        for question in list_questions[1:]:
            payload = {"question": question, "uid_conversation": uid_conversation}
            response = client.post(base_url + "/conv", json=payload, params=query)
            assert response.status_code == 200

            response_json = response.json()
            ai_answer = response_json.get("ai_answer")
            assert ai_answer
            agent_state = response_json.get("agent_state")
            if debug:
                assert agent_state

            print()
            print(f"{question=}")
            print(f"{ai_answer=}")

        print()
        for i, message in enumerate(agent_state.get("messages")):
            if i % 2:
                speaker = "Вы: "
            else:
                speaker = "AI: "

            print(f"{speaker}{message.get('content')}")

        assert EXPECTED_EOF in ai_answer
