"""
Tests for the FastAPI app: the / page (which now serves the frontend)
and the /ask endpoint. We patch (temporarily replace) ask_claude before
each request, so these tests never make a real call to the Claude API —
no API key needed, no cost, no flakiness.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_root_serves_the_frontend_page():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "FAQ Chatbot" in response.text


def test_ask_returns_200_and_the_answer():
    with patch("api.ask_claude", return_value="This is a fake answer."):
        response = client.post("/ask", json={"question": "what are your hours"})

    assert response.status_code == 200
    assert response.json() == {"answer": "This is a fake answer."}


def test_ask_rejects_a_request_missing_the_question_field():
    # No mocking needed here — FastAPI rejects this before our code
    # (and therefore before ask_claude) ever runs, since the request
    # body doesn't match the AskRequest shape.
    response = client.post("/ask", json={})

    assert response.status_code == 422


def test_ask_forwards_conversation_history_to_claude():
    with patch("api.ask_claude", return_value="fake reply") as mock_ask_claude:
        response = client.post(
            "/ask",
            json={
                "question": "how long is that warranty",
                "history": [
                    {"role": "user", "content": "what is your warranty policy"},
                    {
                        "role": "assistant",
                        "content": "1 year warranty against defects.",
                    },
                ],
            },
        )

    assert response.status_code == 200

    # ask_claude is called as ask_claude(client, system_prompt, messages) —
    # check that the third positional argument is the full conversation,
    # not just the newest question.
    messages_sent_to_claude = mock_ask_claude.call_args[0][2]
    assert len(messages_sent_to_claude) == 3
    assert messages_sent_to_claude[0]["content"] == "what is your warranty policy"
    assert messages_sent_to_claude[-1]["content"] == "how long is that warranty"


def test_ask_still_works_with_no_history_field():
    # Confirms backward compatibility: a caller that only sends
    # "question" (like our earlier tests, or any older client) still
    # works — "history" defaults to an empty list.
    with patch("api.ask_claude", return_value="fake reply") as mock_ask_claude:
        response = client.post("/ask", json={"question": "what are your hours"})

    assert response.status_code == 200
    messages_sent_to_claude = mock_ask_claude.call_args[0][2]
    assert messages_sent_to_claude == [
        {"role": "user", "content": "what are your hours"}
    ]
