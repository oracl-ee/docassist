import numpy as np


def top_k_python(vectors, query, k=4):
    """Pure Python: vectors is a list of lists, query is a list."""
    scores = []
    for i, vec in enumerate(vectors):
        s = 0.0
        for a, b in zip(vec, query):
            s += a * b
        scores.append((i, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def top_k_numpy(vectors, query, k=4):
    scores = vectors @ query
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


# GPU search: vectors are copied to GPU memory once and kept there;
# only the small query vector moves on each search.
_gpu = {"src": None, "tensor": None}


def to_gpu(vectors):
    import torch
    if _gpu["src"] is not vectors:
        _gpu["tensor"] = torch.from_numpy(vectors).to("cuda")
        _gpu["src"] = vectors
    return _gpu["tensor"]


def top_k_gpu(vectors, query, k=4):
    import torch
    V = to_gpu(vectors)
    q = torch.from_numpy(query).to("cuda")
    vals, idx = torch.topk(V @ q, min(k, V.shape[0]))
    return list(zip(idx.tolist(), vals.tolist()))


def search(vectors, query, k=4, backend="numpy"):
    if backend in ("cpp", "cpp_mt"):
        import fastsearch
        fn = fastsearch.top_k_parallel if backend == "cpp_mt" else fastsearch.top_k
        return fn(vectors, query, k)
    if backend == "gpu":
        return top_k_gpu(vectors, query, k)
    if backend == "python":
        return top_k_python(vectors.tolist(), query.tolist(), k)
    return top_k_numpy(vectors, query, k)