"""
For each TREC year (2021, 2022) and each embedding model (general, biomed),
retrieve the top-100 candidate trials per patient topic. 2021 and 2022 are
kept fully separate throughout -- 2021 is the dev set, 2022 the held-out
test set -- since both years' topic IDs start at 1 and must never be merged
without namespacing.

    python src/retrieval/retrieve.py

Output: data/processed/retrieval/results_{year}_{model}.csv
        columns: query_id, rank, nct_id, score
"""
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

EMB_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "embeddings"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "retrieval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_KEYS = ["general", "biomed"]
YEARS = [2021, 2022]
TOP_K = 100


def main():
    corpus_ids = pd.read_csv(EMB_DIR / "corpus_id_order.csv")["nct_id"].tolist()

    for key in MODEL_KEYS:
        index = faiss.read_index(str(EMB_DIR / f"corpus_index_{key}.faiss"))
        assert index.ntotal == len(corpus_ids), (
            f"[{key}] index size {index.ntotal} != corpus id list {len(corpus_ids)} "
            f"-- re-run embed_corpus.py / build_index.py, something is out of sync."
        )

        for year in YEARS:
            topic_ids = pd.read_csv(EMB_DIR / f"topics_{year}_id_order.csv")["query_id"].tolist()
            q_emb = np.load(EMB_DIR / f"topics_{year}_{key}.npy").astype("float32")
            assert len(topic_ids) == q_emb.shape[0]

            scores, idxs = index.search(q_emb, TOP_K)

            rows = []
            for qi, (score_row, idx_row) in enumerate(zip(scores, idxs)):
                for rank, (score, corpus_idx) in enumerate(zip(score_row, idx_row), start=1):
                    if corpus_idx == -1:
                        continue
                    rows.append({
                        "query_id": topic_ids[qi],
                        "rank": rank,
                        "nct_id": corpus_ids[corpus_idx],
                        "score": float(score),
                    })

            out = pd.DataFrame(rows)
            out.to_csv(OUT_DIR / f"results_{year}_{key}.csv", index=False)
            print(f"[{key}][{year}] {len(topic_ids)} topics x top-{TOP_K} "
                  f"-> results_{year}_{key}.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
