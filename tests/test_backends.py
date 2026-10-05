import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fastsearch
from app.search import top_k_gpu, top_k_numpy

rng = np.random.default_rng(0)
vecs = rng.standard_normal((20000, 768), dtype=np.float32)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
q = vecs[42]

print("numpy: ", [i for i, _ in top_k_numpy(vecs, q, 4)])
print("cpp:   ", [i for i, _ in fastsearch.top_k(vecs, q, 4)])
print("cpp_mt:", [i for i, _ in fastsearch.top_k_parallel(vecs, q, 4)])
print("gpu:   ", [i for i, _ in top_k_gpu(vecs, q, 4)])