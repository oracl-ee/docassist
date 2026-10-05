import ollama

MODEL = "llama3.1:8b"
KEEP_ALIVE = "30m"  # keep the model in VRAM for 30  minutes

SYSTEM = (
    "Answer the question using only the provided context. "
    "If the answer is not in the context, say you don't know. "
    "Cite sources in brackets, like [notes.pdf]."
)


def build_messages(question, hits, chunks):
    context = "\n\n".join(
        f"[{chunks[i]['source']}]\n{chunks[i]['text']}" for i, _ in hits
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def warm_up():
    """Load the model into VRAM once, so the first real question is fast."""
    ollama.generate(model=MODEL, prompt="", keep_alive=KEEP_ALIVE)


def answer_stream(question, hits, chunks):
    """Yield the answer piece by piece (used by the web app)."""
    stream = ollama.chat(model=MODEL, messages=build_messages(question, hits, chunks),
                         stream=True, keep_alive=KEEP_ALIVE)
    for part in stream:
        yield part["message"]["content"]


def answer(question, hits, chunks):
    """Print the answer as it streams (used by the command-line app)."""
    full = ""
    for piece in answer_stream(question, hits, chunks):
        print(piece, end="", flush=True)
        full += piece
    print()
    return full