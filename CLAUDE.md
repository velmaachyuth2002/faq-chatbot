I'm building an "AI FAQ Chatbot" — a chatbot that answers frequently asked
questions using an LLM, starting from plain hardcoded logic and gradually
adding real AI, retrieval, an API, memory, and deployment.

MY BACKGROUND:
I only know Python. I have never used FastAPI, LLM APIs, embeddings, vector
search, Docker, or deployment before. Treat every one of these as brand new.

HOW I WANT TO WORK:
I do NOT want to write the code myself. For every milestone, give me the
complete, ready-to-run code directly — not a skeleton, not TODOs, not
partial snippets I need to fill in. I will copy, run, and test what you
give me.

For every milestone, structure your response like this:
1. What we're building and why (2-3 sentences).
2. The full code, complete and copy-pasteable.
3. A line-by-line or block-by-block explanation of the code: what each
   part does AND why it's written that way (not just "this is a loop" but
   "this is a loop because we need to keep asking for input until the user
   quits").
4. Any new concept/library introduced — explain it in plain English before
   or alongside the code, assuming I've never seen it before.
5. Exact commands to install anything needed (pip install ...).
6. Exact command to run the code.
7. What a successful test looks like — what to type in, what I should see.

Always follow these rules:
1. Only build the milestone I explicitly ask for. Do not add features from
   later milestones.
2. Never assume prior knowledge of anything beyond plain Python syntax.
3. If a milestone changes/replaces code from a previous milestone, show me
   the full updated file again, not just a diff or snippet.
4. If I hit an error, explain what caused it before giving me the fix.
5. Use environment variables for any API keys, and show me exactly how to
   set them up on my machine.
6. At the end of each milestone, the app must run correctly end to end.
7. If I ask something unrelated to the current milestone, answer it fully,
   then return to the milestone sequence.

PROJECT MILESTONES:
1. Plain Python chatbot (hardcoded replies, no AI)
2. Load FAQs from a JSON file instead of hardcoding them
3. Connect a real LLM (Claude API) to generate answers
4. Prompt engineering (answer only from FAQs, say "I don't know" otherwise)
5. Retrieval for larger FAQ sets (embeddings + similarity search)
6. Wrap it in a FastAPI backend (POST /ask endpoint)
7. Add conversation memory (handle follow-up questions)
8. Simple frontend to talk to the API (optional)
9. Dockerize the app
10. Deploy to a cloud host with a public URL

Current milestone: Milestone 1 — Plain Python chatbot
