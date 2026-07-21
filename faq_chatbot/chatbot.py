"""
Milestone 7: The command-line chatbot now remembers the conversation as it
goes, so it can understand follow-up questions like "what about on
weekends?" instead of treating every message as a brand-new, unrelated
question.
"""

import anthropic
from sentence_transformers import SentenceTransformer

from faq_engine import (
    EMBEDDING_MODEL_NAME,
    ask_claude,
    build_system_prompt,
    embed_faqs,
    load_faqs,
    retrieve_relevant_faqs,
)

# Words that end the conversation when typed by the user.
EXIT_COMMANDS = {"quit", "exit", "bye"}


def main():
    faqs = load_faqs()

    print("Loading embedding model (only needed once per run)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    questions, question_embeddings = embed_faqs(embedder, faqs)

    # Reads your API key automatically from the ANTHROPIC_API_KEY
    # environment variable.
    client = anthropic.Anthropic()

    # Holds every turn of the conversation so far, in the format the
    # Claude API expects: a list of {"role": ..., "content": ...} dicts.
    # We send this whole list on every request, since the API itself
    # doesn't remember anything between calls — WE have to remind it.
    conversation_history = []

    print("FAQ Chatbot (type 'quit' to exit)")
    print(f"Ask me about any of our {len(faqs)} FAQs!\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in EXIT_COMMANDS:
            print("Bot: Goodbye!")
            break

        if user_input == "":
            print("Bot: Please type a question.")
            continue

        # Retrieval still looks only at the latest message, not the whole
        # conversation — that's a simplification worth knowing about (see
        # the milestone write-up for a real example of where it matters).
        relevant_faqs = retrieve_relevant_faqs(
            embedder, user_input, questions, question_embeddings, faqs
        )
        system_prompt = build_system_prompt(relevant_faqs)

        conversation_history.append({"role": "user", "content": user_input})
        answer = ask_claude(client, system_prompt, conversation_history)
        conversation_history.append({"role": "assistant", "content": answer})

        print(f"Bot: {answer}")


if __name__ == "__main__":
    main()
