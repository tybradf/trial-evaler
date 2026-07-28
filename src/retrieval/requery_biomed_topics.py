"""
Re-embed ONLY the 2021/2022 topic queries for the biomed (MedCPT) model,
using the corrected max_length=512. The corpus embeddings and index (from
the article encoder, unaffected by this bug) don't need to be touched.

    python src/retrieval/requery_biomed_topics.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from embedders import MedCPTEmbedder

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
EMB_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "embeddings"


def main():
    embedder = MedCPTEmbedder()
    for year in [2021, 2022]:
        df = pd.read_csv(RAW_DIR / f"topics_{year}.csv")
        print(f"[biomed] re-embedding {year} topics ({len(df)}) with max_length=512 ...")
        q_emb = embedder.encode_queries(df["text"].tolist())
        np.save(EMB_DIR / f"topics_{year}_biomed.npy", q_emb)
        print(f"[biomed] {year} topic embeddings overwritten, shape={q_emb.shape}")


if __name__ == "__main__":
    main()
