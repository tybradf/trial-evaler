"""
Score both embedding models' retrieval output against the physician qrels,
for 2021 (dev) and 2022 (held-out test) separately.

Metrics reported (all standard IR conventions, documents not in qrels are
treated as non-relevant per TREC convention):

  recall_eligible@k    -- relevant = relevance==2 (eligible only)
  recall_relevant@k     -- relevant = relevance>=1 (excluded OR eligible;
                            i.e. "on-topic", the trial at least matches the
                            patient's condition even if they don't qualify)
  ndcg@10               -- standard graded nDCG, exponential gain
                            (2^rel - 1)/log2(rank+1), rel in {0,1,2}

Reporting both recall variants rather than picking one: "eligible" is the
stricter, arguably more clinically meaningful bar; "relevant" is the
looser one closer to classic ad-hoc retrieval. Worth deciding which to
headline once you see both -- I'd lean eligible for the writeup since
that's what actually matters for trial matching, but relevant is useful
context for how well the retriever finds the right *disease area* even
when it's wrong about eligibility specifically.

    python src/eval/retrieval_metrics.py
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RETR_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "retrieval"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "retrieval_metrics.csv"

MODEL_KEYS = ["general", "biomed"]
YEARS = [2021, 2022]
RECALL_KS = [10, 50, 100]


def load_qrels(year: int) -> dict:
    df = pd.read_csv(RAW_DIR / f"qrels_{year}.csv")
    qrels = defaultdict(dict)
    for row in df.itertuples():
        qrels[row.query_id][row.doc_id] = int(row.relevance)
    return qrels


def ndcg_at_10(ranked_nct_ids: list[str], rel_lookup: dict) -> float:
    def gain(rel):
        return (2 ** rel) - 1

    dcg = sum(
        gain(rel_lookup.get(nct_id, 0)) / np.log2(rank + 1)
        for rank, nct_id in enumerate(ranked_nct_ids[:10], start=1)
    )
    ideal_rels = sorted(rel_lookup.values(), reverse=True)[:10]
    idcg = sum(gain(rel) / np.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_nct_ids: list[str], rel_lookup: dict, k: int, threshold: int) -> float:
    relevant_ids = {nct for nct, rel in rel_lookup.items() if rel >= threshold}
    if not relevant_ids:
        return np.nan  # topic has no relevant docs at this threshold -- exclude, don't zero it out
    retrieved_topk = set(ranked_nct_ids[:k])
    return len(retrieved_topk & relevant_ids) / len(relevant_ids)


def main():
    summary_rows = []

    for year in YEARS:
        qrels = load_qrels(year)

        for key in MODEL_KEYS:
            results = pd.read_csv(RETR_DIR / f"results_{year}_{key}.csv")
            results = results.sort_values(["query_id", "rank"])

            per_topic_metrics = []
            for query_id, group in results.groupby("query_id"):
                ranked = group.sort_values("rank")["nct_id"].tolist()
                rel_lookup = qrels.get(query_id, {})

                row = {"query_id": query_id, "ndcg@10": ndcg_at_10(ranked, rel_lookup)}
                for k in RECALL_KS:
                    row[f"recall_eligible@{k}"] = recall_at_k(ranked, rel_lookup, k, threshold=2)
                    row[f"recall_relevant@{k}"] = recall_at_k(ranked, rel_lookup, k, threshold=1)
                per_topic_metrics.append(row)

            per_topic_df = pd.DataFrame(per_topic_metrics)
            means = per_topic_df.drop(columns="query_id").mean(numeric_only=True)

            summary_rows.append({"year": year, "model": key, **means.to_dict()})
            print(f"\n[{year}][{key}]  n_topics={len(per_topic_df)}")
            print(means.round(4).to_string())

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_PATH, index=False)
    print(f"\nSummary saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
