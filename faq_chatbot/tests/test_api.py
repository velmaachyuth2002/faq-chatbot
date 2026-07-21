"""
Tests for the FastAPI /ask endpoint. We patch (temporarily replace)
ask_claude before each request, so these tests never make a real call
to the Claude API — no API key needed, no cost, no flakiness.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


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
