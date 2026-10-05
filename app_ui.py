"""DocAssist web app: drop in documents, then ask questions about them.
"""
import time
from collections import Counter
from pathlib import Path

import numpy as np
import streamlit as st

from app.chunker import chunk_documents
from app.embedder import embed_documents, embed_query
from app.llm import MODEL, answer_stream, warm_up
from app.loader import load_file
from app.search import search
from app.store import index_exists, load_index, save_index

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
EMBED_DIM = 768

st.set_page_config(page_title="DocAssist", page_icon="📄", layout="wide")


# ---------- index helpers ----------

@st.cache_resource
def start_model():
    warm_up()  # load the LLM into VRAM once per server start
    return True


def get_index():
    if "vectors" not in st.session_state:
        if index_exists():
            vectors, chunks = load_index()
        else:
            vectors, chunks = np.zeros((0, EMBED_DIM), dtype=np.float32), []
        st.session_state.vectors = vectors
        st.session_state.chunks = chunks
    return st.session_state.vectors, st.session_state.chunks


def set_index(vectors, chunks):
    save_index(vectors, chunks)
    st.session_state.vectors = vectors
    st.session_state.chunks = chunks


def without_source(vectors, chunks, name):
    keep = [i for i, c in enumerate(chunks) if c["source"] != name]
    return vectors[keep], [chunks[i] for i in keep]


def add_files(files, bar):
    """Save uploads to data/, then embed only the new files and append them."""
    vectors, chunks = get_index()
    added, skipped = [], []
    for n, f in enumerate(files, 1):
        bar.progress((n - 1) / len(files), text=f"Reading {f.name}")
        path = DATA_DIR / f.name
        path.write_bytes(f.getbuffer())
        doc = load_file(path)
        if doc is None:
            path.unlink(missing_ok=True)
            skipped.append(f.name)
            continue
        vectors, chunks = without_source(vectors, chunks, f.name)  # replace old version
        new_chunks = chunk_documents([doc])
        bar.progress((n - 0.5) / len(files), text=f"Embedding {f.name} ({len(new_chunks)} chunks)")
        new_vectors = embed_documents([c["text"] for c in new_chunks])
        vectors = np.vstack([vectors, new_vectors])
        chunks = chunks + new_chunks
        added.append(f"{f.name} ({len(new_chunks)} chunks)")
    bar.progress(1.0, text="Done")
    set_index(vectors, chunks)
    return added, skipped


def remove_file(name):
    vectors, chunks = get_index()
    set_index(*without_source(vectors, chunks, name))
    (DATA_DIR / name).unlink(missing_ok=True)


# ---------- page ----------

start_model()
vectors, chunks = get_index()

for key, default in (("uploader_key", 0), ("messages", []), ("notice", None)):
    st.session_state.setdefault(key, default)

with st.sidebar:
    st.header("Your documents")
    files = st.file_uploader(
        "Drop PDF, .txt, or .md files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if files and st.button(f"Add {len(files)} file(s) to library", type="primary",
                           use_container_width=True):
        bar = st.progress(0.0, text="Starting")
        added, skipped = add_files(files, bar)
        msg = f"Added: {', '.join(added)}." if added else ""
        if skipped:
            msg += f" Skipped (no readable text): {', '.join(skipped)}."
        st.session_state.notice = msg.strip()
        st.session_state.uploader_key += 1  # clears the drop zone
        st.rerun()

    if st.session_state.notice:
        st.success(st.session_state.notice)
        st.session_state.notice = None

    counts = Counter(c["source"] for c in chunks)
    if counts:
        st.caption(f"{len(counts)} documents, {len(chunks)} chunks indexed")
        for name, n in sorted(counts.items()):
            col1, col2 = st.columns([4, 1])
            col1.write(f"{name}  \n:gray[{n} chunks]")
            if col2.button("✕", key=f"rm_{name}", help=f"Remove {name}"):
                remove_file(name)
                st.rerun()
    else:
        st.info("No documents yet. Drop a file above to start.")

    st.divider()
    st.subheader("Search settings")
    backend = st.selectbox("Search backend", ["numpy", "cpp", "cpp_mt", "gpu", "python"],
                           help="Same results, different speed. Compare them!")
    k = st.slider("Chunks sent to the model", 2, 8, 4)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("DocAssist")
st.write(f"Ask questions about your documents. Answers come from {MODEL} running "
         "on your own GPU, and every answer cites its sources.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("caption"):
            st.caption(m["caption"])

question = st.chat_input("Ask a question about your documents" if chunks
                         else "Add a document in the sidebar first", disabled=not chunks)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        q = embed_query(question)
        start = time.perf_counter()
        try:
            hits = search(vectors, q, k, backend)
        except ImportError:
            st.error("The C++ module isn't built yet. Run: python setup.py build_ext --inplace")
            st.stop()
        except Exception as e:  # e.g. GPU backend without CUDA
            st.error(f"{backend} search failed: {e}")
            st.stop()
        ms = (time.perf_counter() - start) * 1000

        with st.expander("Sources used"):
            for i, score in hits:
                c = chunks[i]
                st.markdown(f"**{c['source']}**, chunk {c['chunk_id']} (similarity {score:.3f})")
                st.caption(c["text"][:300] + "...")

        reply = st.write_stream(answer_stream(question, hits, chunks))
        sources = sorted({chunks[i]["source"] for i, _ in hits})
        caption = f"Sources: {', '.join(sources)}. {backend} search took {ms:.2f} ms."
        st.caption(caption)

    st.session_state.messages.append({"role": "assistant", "content": reply, "caption": caption})