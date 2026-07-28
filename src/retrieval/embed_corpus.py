"""
Embed the trial corpus once (shared across both TREC years) and the 2021 +
2022 patient topics separately, with both the general-purpose and biomedical
embedders. Saves raw numpy arrays + the id ordering each array corresponds
to, so build_index.py / retrieve.py can load them without re-embedding.

    python src/retrieval/embed_corpus.py

Runtime note: ~48.7k trials on CPU will take a while (expect tens of minutes
per model, less on Apple Silicon MPS or a CUDA GPU -- DEVICE auto-detects in
embedders.py). Topics (125 total) are effectively instant. If you want to
sanity-check the pipeline before committing to the full run, temporarily
slice `corpus = corpus.head(500)` below.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from embedders import EMBEDDERS

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
EMB_DIR = PROCESSED_DIR / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

BODY_CHAR_CAP = 4000  # courtesy cap on outlier-long trials before tokenization


def _safe_str(x) -> str:
    return x if isinstance(x, str) else ""


def build_corpus_body(row) -> str:
    parts = [_safe_str(row.get("conditions")), _safe_str(row.get("brief_summary")),
             _safe_str(row.get("eligibility_raw"))]
    return " ".join(p for p in parts if p)[:BODY_CHAR_CAP]


def main():
    corpus = pd.read_parquet(PROCESSED_DIR / "corpus.parquet")
    corpus = corpus.dropna(subset=["nct_id"]).reset_index(drop=True)
    titles = corpus["brief_title"].fillna("").astype(str).tolist()
    bodies = corpus.apply(build_corpus_body, axis=1).tolist()
    nct_ids = corpus["nct_id"].tolist()

    # Save id ordering once -- both models' corpus embeddings share this order.
    pd.Series(nct_ids).to_csv(EMB_DIR / "corpus_id_order.csv", index=False, header=["nct_id"])

    topics = {
        2021: pd.read_csv(RAW_DIR / "topics_2021.csv"),
        2022: pd.read_csv(RAW_DIR / "topics_2022.csv"),
    }
    for year, df in topics.items():
        df["query_id"].to_csv(EMB_DIR / f"topics_{year}_id_order.csv", index=False)

    for key, cls in EMBEDDERS.items():
        print(f"\n=== {key} ===")
        embedder = cls()

        print(f"[{key}] embedding corpus ({len(nct_ids)} trials) ...")
        corpus_emb = embedder.encode_docs(titles, bodies)
        np.save(EMB_DIR / f"corpus_{key}.npy", corpus_emb)
        print(f"[{key}] corpus embeddings saved, shape={corpus_emb.shape}")

        for year, df in topics.items():
            print(f"[{key}] embedding {year} topics ({len(df)}) ...")
            q_emb = embedder.encode_queries(df["text"].tolist())
            np.save(EMB_DIR / f"topics_{year}_{key}.npy", q_emb)
            print(f"[{key}] {year} topic embeddings saved, shape={q_emb.shape}")

        # Free memory before loading the next model.
        del embedder

    print(f"\nAll embeddings saved to {EMB_DIR}")


if __name__ == "__main__":
    main()
