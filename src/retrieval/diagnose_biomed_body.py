"""
Controlled test: does feeding MedCPT's article encoder raw bulleted
eligibility criteria (a format it never saw in training) hurt match
quality/discriminability compared to a cleaner title+summary body closer
to its PubMed abstract training distribution?

Uses a RANDOM sample (not .head()) to avoid the chronological-sort confound
in the earlier 100-trial smoke test.

    python src/retrieval/diagnose_biomed_body.py --n_sample 300 --n_topics 3
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from embedders import MedCPTEmbedder

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _safe_str(x) -> str:
    return x if isinstance(x, str) else ""


def body_full(row) -> str:
    parts = [_safe_str(row.get("conditions")), _safe_str(row.get("brief_summary")),
             _safe_str(row.get("eligibility_raw"))]
    return " ".join(p for p in parts if p)[:4000]


def body_summary_only(row) -> str:
    parts = [_safe_str(row.get("conditions")), _safe_str(row.get("brief_summary"))]
    return " ".join(p for p in parts if p)[:2000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sample", type=int, default=300)
    ap.add_argument("--n_topics", type=int, default=3)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--year", type=int, default=2021)
    args = ap.parse_args()

    corpus = pd.read_parquet(PROCESSED_DIR / "corpus.parquet").dropna(subset=["nct_id"])
    sample = corpus.sample(n=args.n_sample, random_state=42).reset_index(drop=True)
    titles = sample["brief_title"].fillna("").astype(str).tolist()
    nct_ids = sample["nct_id"].tolist()

    topics = pd.read_csv(RAW_DIR / f"topics_{args.year}.csv").head(args.n_topics)

    print(f"Random sample of {len(sample)} trials (not chronologically biased)\n")
    embedder = MedCPTEmbedder()
    q_emb = embedder.encode_queries(topics["text"].tolist())

    variants = {
        "A: full (title + summary + raw eligibility)": sample.apply(body_full, axis=1).tolist(),
        "B: summary-only (title + summary, no eligibility dump)": sample.apply(body_summary_only, axis=1).tolist(),
    }

    for label, bodies in variants.items():
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        doc_emb = embedder.encode_docs(titles, bodies)
        sims_all = doc_emb @ q_emb.T  # (n_docs, n_topics)
        print(f"Score range across all topic-doc pairs: "
              f"min={sims_all.min():.3f}  max={sims_all.max():.3f}  "
              f"std={sims_all.std():.3f}  (higher std = more discriminating)")

        for i, row in topics.reset_index(drop=True).iterrows():
            print(f"\n--- topic {row['query_id']} ---")
            print(row["text"][:200] + "...")
            sims = sims_all[:, i]
            top_idx = np.argsort(-sims)[:args.top_k]
            for rank, idx in enumerate(top_idx, start=1):
                meta = sample.iloc[idx]
                print(f"  {rank}. [{sims[idx]:.3f}] {nct_ids[idx]} -- "
                      f"{meta['brief_title']}  ({meta['conditions']})")


if __name__ == "__main__":
    main()
