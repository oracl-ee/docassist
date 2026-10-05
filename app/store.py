import json
from pathlib import Path

import numpy as np

INDEX_DIR = Path("index")


def save_index(vectors, chunks):
    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")


def load_index():
    vectors = np.load(INDEX_DIR / "vectors.npy")
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    return vectors, chunks


def index_exists():
    return (INDEX_DIR / "vectors.npy").exists()