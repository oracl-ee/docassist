import numpy as np
import ollama
MODEL = "nomic-embed-text"
def _normalize(arr):
norms = np.linalg.norm(arr, axis=1, keepdims=True)
return arr / np.maximum(norms, 1e-10) # avoid dividing by zero
def embed_documents(texts, batch_size=32):
vectors = []
for i in range(0, len(texts), batch_size):
batch = ["search_document: " + t for t in texts[i:i + batch_size]]
vectors.extend(ollama.embed(model=MODEL, input=batch)["embeddings"])
print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")
return _normalize(np.array(vectors, dtype=np.float32))
def embed_query(text):
vec = ollama.embed(model=MODEL, input="search_query: " +