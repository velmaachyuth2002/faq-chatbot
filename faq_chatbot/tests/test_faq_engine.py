"""
Tests for faq_engine.py — the shared FAQ-answering logic. These tests
never call the real Claude API: ask_claude is tested with a fake client
instead of a real one, so the whole suite runs for free, needs no API
key, and always gets the same predictable result instead of depending
on whatever Claude feels like saying that day.
"""

from unittest.mock import MagicMock

from sentence_transformers import SentenceTransformer

from faq_engine import (
    EMBEDDING_MODEL_NAME,
    NO_ANSWER_MESSAGE,
    ask_claude,
    build_system_prompt,
    cosine_similarity,
    embed_faqs,
    load_faqs,
    retrieve_relevant_faqs,
)


def test_load_faqs_returns_a_non_empty_dict():
    faqs = load_faqs()
    assert isinstance(faqs, dict)
    assert len(faqs) > 0


def test_load_faqs_contains_an_expected_question():
    faqs = load_faqs()
    assert "What are your business hours?" in faqs


def test_cosine_similarity_of_identical_vectors_is_one():
    vector = [1.0, 2.0, 3.0]
    assert cosine_similarity(vector, vector) == 1.0


def test_cosine_similarity_of_opposite_vectors_is_negative_one():
    vector_a = [1.0, 0.0]
    vector_b = [-1.0, 0.0]
    assert cosine_similarity(vector_a, vector_b) == -1.0


def test_cosine_similarity_of_unrelated_vectors_is_zero():
    vector_a = [1.0, 0.0]
    vector_b = [0.0, 1.0]
    assert cosine_similarity(vector_a, vector_b) == 0.0


def test_build_system_prompt_includes_the_faq_content():
    relevant_faqs = [("What are your hours?", "9 AM to 5 PM.")]

    prompt = build_system_prompt(relevant_faqs)

    assert "What are your hours?" in prompt
    assert "9 AM to 5 PM." in prompt
    assert NO_ANSWER_MESSAGE in prompt


def test_retrieve_relevant_faqs_finds_the_right_match():
    # This test loads the real embedding model (no API key needed, just
    # the same local model we use everywhere else) to prove retrieval
    # actually works end to end, not just that the function runs.
    faqs = load_faqs()
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    questions, embeddings = embed_faqs(embedder, faqs)

    results = retrieve_relevant_faqs(
        embedder, "how long is the warranty", questions, embeddings, faqs
    )

    matched_questions = [question for question, _ in results]
    assert "What is your warranty policy?" in matched_questions


def test_ask_claude_returns_claudes_text_response():
    # Build a fake ("mock") Claude client instead of a real one. A real
    # client would make a real network request to Anthropic's servers —
    # costing money and needing a valid API key. A fake client lets us
    # test "does ask_claude correctly pull the text out of the
    # response?" without any of that.
    fake_text_block = MagicMock()
    fake_text_block.type = "text"
    fake_text_block.text = "This is a fake answer."

    fake_response = MagicMock()
    fake_response.content = [fake_text_block]

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    answer = ask_claude(
        fake_client, "system prompt", [{"role": "user", "content": "hi"}]
    )

    assert answer == "This is a fake answer."
