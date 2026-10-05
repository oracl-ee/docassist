import argparse
import time

from app.chunker import chunk_documents
from app.embedder import embed_documents, embed_query
from app.llm import answer
from app.loader import load_documents
from app.search import search
from app.store import index_exists, load_index, save_index

BACKENDS = ["python", "numpy", "cpp", "cpp_mt", "gpu"]


def build_index():
    docs = load_documents()
    if not docs:
        print("No documents found in data/")
        return
    chunks = chunk_documents(docs)
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    vectors = embed_documents([c["text"] for c in chunks])
    save_index(vectors, chunks)
    print("Index saved to index/")


def ask(question, vectors, chunks, backend, k):
    q = embed_query(question)
    start = time.perf_counter()
    hits = search(vectors, q, k, backend)
    ms = (time.perf_counter() - start) * 1000
    print(f"\nTop {k} chunks ({backend} search, {ms:.2f} ms):")
    for i, score in hits:
        print(f"  {chunks[i]['source']} #{chunks[i]['chunk_id']}  score={score:.3f}")
    print("\nAnswer:")
    answer(question, hits, chunks)


def main():
    parser = argparse.ArgumentParser(description="DocAssist: ask questions about your documents")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="Build the index from data/")
    ask_p = sub.add_parser("ask", help="Ask one question")
    ask_p.add_argument("question")
    chat_p = sub.add_parser("chat", help="Ask questions in a loop")
    for p in (ask_p, chat_p):
        p.add_argument("--backend", choices=BACKENDS, default="numpy")
        p.add_argument("-k", type=int, default=4)
    args = parser.parse_args()

    if args.command == "index":
        build_index()
        return
    if not index_exists():
        print("No index yet. Upload files in the web app, or run: python main.py index")
        return
    vectors, chunks = load_index()
    if args.command == "ask":
        ask(args.question, vectors, chunks, args.backend, args.k)
    else:
        print("Type a question, or 'quit' to exit.")
        while (q := input("\n> ").strip()).lower() not in ("quit", "exit"):
            if q:
                ask(q, vectors, chunks, args.backend, args.k)


if __name__ == "__main__":
    main()