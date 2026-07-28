"""
Build one FAISS index per embedding model from the corpus embeddings saved
by embed_corpus.py. Vectors are already L2-normalized, so IndexFlatIP (inner
product) is equivalent to cosine similarity ranking. At ~48.7k trials this
is small enough that exact search is instant -- no need for an approximate
index (IVF/HNSW) at this corpus size.

    python src/retrieval/build_index.py
"""
from pathlib import Path

import faiss
import numpy as np

EMB_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "embeddings"
MODEL_KEYS = ["general", "biomed"]


def main():
    for key in MODEL_KEYS:
        emb = np.load(EMB_DIR / f"corpus_{key}.npy").astype("float32")
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        faiss.write_index(index, str(EMB_DIR / f"corpus_index_{key}.faiss"))
        print(f"[{key}] index built: {index.ntotal} vectors, dim={dim} "
              f"-> corpus_index_{key}.faiss")


if __name__ == "__main__":
    main()
