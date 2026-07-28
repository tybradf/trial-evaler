"""
Quick sanity check for a partial embed_corpus.py run (e.g. corpus.head(100)
during development) -- no qrels needed, since a small random corpus slice
almost certainly won't contain the judged trials for any real topic anyway.

Instead: embed a handful of real 2021 patient topics (already done by
embed_corpus.py regardless of corpus size) and rank your N embedded trials
against each one. If the embeddings are working, the top hits should be at
least loosely topical -- e.g. a diabetes patient topic surfacing something
diabetes/endocrine-related, not random noise. This won't validate retrieval
*quality* (that needs the full corpus + qrels), just that the pipeline
isn't broken -- wrong pooling, mismatched id ordering, garbage vectors, etc.

    python src/retrieval/sanity_check.py --n_topics 3 --top_k 5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EMB_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "embeddings"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MODEL_KEYS = ["general", "biomed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--n_topics", type=int, default=3)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    corpus_ids = pd.read_csv(EMB_DIR / "corpus_id_order.csv")["nct_id"].tolist()
    corpus_meta = pd.read_parquet(PROCESSED_DIR / "corpus.parquet").set_index("nct_id")
    topics = pd.read_csv(RAW_DIR / f"topics_{args.year}.csv")

    print(f"Corpus slice size: {len(corpus_ids)} trials\n")

    for key in MODEL_KEYS:
        print(f"\n{'=' * 70}\nMODEL: {key}\n{'=' * 70}")
        corpus_emb = np.load(EMB_DIR / f"corpus_{key}.npy")
        topic_emb = np.load(EMB_DIR / f"topics_{args.year}_{key}.npy")
        topic_ids = pd.read_csv(EMB_DIR / f"topics_{args.year}_id_order.csv")["query_id"].tolist()

        assert corpus_emb.shape[0] == len(corpus_ids), "corpus embedding count != id list"
        assert topic_emb.shape[0] == len(topic_ids), "topic embedding count != id list"

        for i in range(min(args.n_topics, len(topic_ids))):
            qid = topic_ids[i]
            qtext = topics.loc[topics["query_id"] == qid, "text"].iloc[0]
            print(f"\n--- topic {qid} ---")
            print(qtext[:300] + ("..." if len(qtext) > 300 else ""))

            sims = corpus_emb @ topic_emb[i]
            top_idx = np.argsort(-sims)[:args.top_k]
            print(f"\ntop-{args.top_k} matches:")
            for rank, idx in enumerate(top_idx, start=1):
                nct = corpus_ids[idx]
                meta = corpus_meta.loc[nct] if nct in corpus_meta.index else None
                title = meta["brief_title"] if meta is not None else "(not found in corpus.parquet)"
                cond = meta["conditions"] if meta is not None else ""
                print(f"  {rank}. [{sims[idx]:.3f}] {nct} -- {title}  ({cond})")


if __name__ == "__main__":
    main()
