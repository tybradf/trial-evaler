"""
Isolate topics where biomed's nDCG@10 was exactly 0.000 -- a qualitatively
different failure (nothing relevant in top 10 at all) from generally-worse
ranking. Prints full topic text so patterns (phrasing, structure, content
type) are visible, not just summary stats.

    python src/eval/find_biomed_zero_topics.py
"""
from pathlib import Path

import pandas as pd

from retrieval_metrics import ndcg_at_10, load_qrels

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RETR_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "retrieval"
YEARS = [2021, 2022]


def main():
    for year in YEARS:
        qrels = load_qrels(year)
        topics = pd.read_csv(RAW_DIR / f"topics_{year}.csv").set_index("query_id")

        biomed = pd.read_csv(RETR_DIR / f"results_{year}_biomed.csv")
        general = pd.read_csv(RETR_DIR / f"results_{year}_general.csv")

        zero_topics = []
        for query_id, group in biomed.groupby("query_id"):
            ranked = group.sort_values("rank")["nct_id"].tolist()
            rel_lookup = qrels.get(query_id, {})
            if ndcg_at_10(ranked, rel_lookup) == 0.0:
                g_ranked = general[general["query_id"] == query_id].sort_values("rank")["nct_id"].tolist()
                g_ndcg = ndcg_at_10(g_ranked, rel_lookup)
                zero_topics.append((query_id, g_ndcg))

        print(f"\n{'=' * 70}\n{year}: {len(zero_topics)} / {biomed['query_id'].nunique()} "
              f"topics scored a hard zero for biomed\n{'=' * 70}")
        for query_id, g_ndcg in zero_topics:
            print(f"\n--- topic {query_id} (biomed=0.000, general={g_ndcg:.3f}) ---")
            print(topics.loc[query_id, "text"])


if __name__ == "__main__":
    main()
