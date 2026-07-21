"""
FastAPI web backend that exposes our FAQ chatbot as an HTTP API, with a
single POST /ask endpoint. Any other program (a website, a mobile app,
another script) can get answers from our chatbot over the network, instead
of needing to run our Python code directly.

Milestone 8 update: added CORS support so the browser-based frontend
(frontend/index.html) is allowed to call this API from a page opened
directly from disk.

Later update: this server now also serves the frontend page itself at
"/", so the one deployed URL works as a complete, shareable chat page —
not just a raw API other developers would call.
"""

from pathlib import Path
from typing import List

import anthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from faq_engine import (
    EMBEDDING_MODEL_NAME,
    ask_claude,
    build_system_prompt,
    embed_faqs,
    load_faqs,
    retrieve_relevant_faqs,
)

# Everything below runs ONCE, when the server starts up — not on every
# request. Loading the embedding model and computing FAQ embeddings is
# relatively slow, so we do it a single time and reuse the results for
# every question that comes in.
faqs = load_faqs()
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
questions, question_embeddings = embed_faqs(embedder, faqs)
client = anthropic.Anthropic()

# Path to the frontend HTML file, so we can serve it directly.
FRONTEND_FILE = Path(__file__).parent / "frontend" / "index.html"

app = FastAPI()

# Browsers block a webpage from calling an API on a different origin
# (different domain/port) unless the API explicitly allows it — this is
# called CORS (Cross-Origin Resource Sharing). Our frontend page will be
# opened directly from a file on disk, which browsers treat as its own
# origin, so without this our JavaScript's fetch() call to the API would
# be silently blocked. allow_origins=["*"] means "allow any origin" —
# fine for local development, but you'd lock this down to your real
# frontend's domain in a production app.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    """
    One turn of a conversation, matching the shape Claude's API itself
    uses: who said it ("user" or "assistant") and what they said.
    """

    role: str
    content: str


class AskRequest(BaseModel):
    """
    Defines the shape of the JSON body we expect on POST /ask.
    FastAPI uses this to validate incoming requests automatically.

    question: the newest thing the user just asked.
    history: every prior turn of this conversation, oldest first. The
    API itself has no memory of its own — the caller (our frontend) is
    responsible for keeping track of the conversation and sending it
    back on every request. Defaults to empty for a brand-new
    conversation, so existing callers that only send "question" still
    work unchanged.
    """

    question: str
    history: List[Message] = []


class AskResponse(BaseModel):
    """
    Defines the shape of the JSON we send back.
    """

    answer: str


@app.get("/")
def serve_frontend() -> FileResponse:
    """
    Serves the chat page itself, so visiting this server's URL directly
    in a browser shows a real chat interface instead of a 404.
    """
    return FileResponse(FRONTEND_FILE)


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    # Retrieval still only looks at the newest question, not the full
    # conversation — the same simplification we made back when we first
    # added conversation memory to the CLI chatbot.
    relevant_faqs = retrieve_relevant_faqs(
        embedder, request.question, questions, question_embeddings, faqs
    )
    system_prompt = build_system_prompt(relevant_faqs)

    messages = [{"role": m.role, "content": m.content} for m in request.history]
    messages.append({"role": "user", "content": request.question})

    answer = ask_claude(client, system_prompt, messages)
    return AskResponse(answer=answer)
