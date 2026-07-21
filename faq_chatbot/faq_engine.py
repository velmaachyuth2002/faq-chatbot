"""
Milestone 6: Shared FAQ-answering logic, used by both the command-line
chatbot (chatbot.py) and the new FastAPI backend (api.py). Pulling this out
into its own module means both interfaces call the exact same code instead
of maintaining two separate, possibly-drifting copies.
"""

import json
from pathlib import Path

import numpy as np

# Build a path to faqs.json based on this module's own location, so it
# loads correctly no matter which script imports this file or where you
# run it from.
FAQS_FILE = Path(__file__).parent / "faqs.json"

# Which Claude model to use for generating answers.
# Haiku is Anthropic's fastest, cheapest model ($1/$5 per 1M tokens vs Opus's
# $5/$25). For an FAQ bot that only answers short questions from text we
# provide, Haiku is a great fit and cuts the cost per question by ~5x.
MODEL = "claude-haiku-4-5"

# Which local embedding model to use for retrieval.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# How many of the most relevant FAQs to hand to Claude per question.
TOP_K = 3

# What Claude should say when a question isn't covered by our FAQs.
NO_ANSWER_MESSAGE = "I don't know. Please contact our support team for help with that."


def load_faqs(filepath=FAQS_FILE):
    """
    Open the JSON file and turn its contents into a Python dictionary.
    """
    with open(filepath, "r") as f:
        return json.load(f)


def embed_faqs(embedder, faqs):
    """
    Compute an embedding vector for every FAQ question. We keep the
    questions and their embeddings in two matching lists (same order,
    same index) so we can look one up by position later.
    """
    questions = list(faqs.keys())
    embeddings = embedder.encode(questions)
    return questions, embeddings


def cosine_similarity(vector_a, vector_b):
    """
    Measure how similar two vectors are, from -1 (opposite meaning)
    to 1 (identical meaning). 0 means unrelated.
    """
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def retrieve_relevant_faqs(embedder, user_input, questions, question_embeddings, faqs):
    """
    Embed the user's question, compare it against every FAQ's embedding,
    and return the TOP_K most similar FAQs as a list of (question, answer)
    pairs, best match first.
    """
    query_embedding = embedder.encode(user_input)

    similarities = [
        cosine_similarity(query_embedding, faq_embedding)
        for faq_embedding in question_embeddings
    ]

    # np.argsort returns the indices that would sort similarities from
    # lowest to highest. We take the last TOP_K (the highest scores) and
    # reverse them so the best match comes first.
    best_indices = np.argsort(similarities)[-TOP_K:][::-1]

    return [(questions[i], faqs[questions[i]]) for i in best_indices]


def build_system_prompt(relevant_faqs):
    """
    Turn a list of (question, answer) pairs into a block of text we can
    hand to Claude as background knowledge, plus strict instructions to
    only answer from that information.
    """
    faq_lines = [f"- {question}: {answer}" for question, answer in relevant_faqs]
    faq_text = "\n".join(faq_lines)

    return (
        "You are a customer support assistant. Answer the user's question "
        "using ONLY the FAQ information listed below. Do not use any "
        "outside knowledge, even if you happen to know the answer.\n\n"
        "If the question cannot be answered using the FAQ information "
        f'below, respond with exactly this sentence: "{NO_ANSWER_MESSAGE}"\n\n'
        "FAQ information:\n"
        f"{faq_text}"
    )


def ask_claude(client, system_prompt, messages):
    """
    Send the conversation so far to Claude, along with our FAQ system
    prompt, and return Claude's newest generated answer as a string.

    `messages` is a list of {"role": ..., "content": ...} turns — the same
    format the Claude API itself uses. Passing the full history (not just
    the latest question) is what lets Claude understand follow-up
    questions like "what about on weekends?".
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=system_prompt,
        messages=messages,
    )

    for block in response.content:
        if block.type == "text":
            return block.text

    return "(Claude didn't return any text.)"
