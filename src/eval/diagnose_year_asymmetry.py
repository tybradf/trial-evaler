"""
Investigate the 2021-vs-2022 asymmetry in biomed's post-fix improvement.
Checks two candidate explanations against per-topic nDCG@10:

  1. Topic length -- do longer narratives correlate with worse biomed
     scores (embedding dilution), and does general show the same pattern
     or not (if only biomed correlates, it's model-specific; if both do,
     it's more likely an intrinsic topic-difficulty effect)?
  2. Qrels density -- topics with very few judged-relevant trials produce
     noisier per-topic recall/nDCG regardless of retrieval quality; if
     2021 topics systematically have sparser judgments, that alone could
     explain part of the gap.

    python src/eval/diagnose_year_asymmetry.py
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from retrieval_metrics import ndcg_at_10, load_qrels  # reuse the exact same eval logic

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RETR_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "retrieval"

YEARS = [2021, 2022]
MODEL_KEYS = ["general", "biomed"]


def per_topic_ndcg(year: int, key: str, qrels: dict) -> pd.DataFrame:
    results = pd.read_csv(RETR_DIR / f"results_{year}_{key}.csv")
    rows = []
    for query_id, group in results.groupby("query_id"):
        ranked = group.sort_values("rank")["nct_id"].tolist()
        rel_lookup = qrels.get(query_id, {})
        n_relevant = sum(1 for r in rel_lookup.values() if r >= 1)
        n_eligible = sum(1 for r in rel_lookup.values() if r == 2)
        rows.append({
            "query_id": query_id,
            f"ndcg_{key}": ndcg_at_10(ranked, rel_lookup),
            "n_relevant_judged": n_relevant,
            "n_eligible_judged": n_eligible,
        })
    return pd.DataFrame(rows)


def main():
    for year in YEARS:
        print(f"\n{'=' * 60}\n{year}\n{'=' * 60}")
        qrels = load_qrels(year)
        topics = pd.read_csv(RAW_DIR / f"topics_{year}.csv")
        topics["word_count"] = topics["text"].str.split().apply(len)

        general_df = per_topic_ndcg(year, "general", qrels)
        biomed_df = per_topic_ndcg(year, "biomed", qrels)

        merged = topics.merge(general_df, on="query_id").merge(
            biomed_df[["query_id", "ndcg_biomed"]], on="query_id"
        )

        print(f"topic word count: mean={merged['word_count'].mean():.0f}  "
              f"median={merged['word_count'].median():.0f}")
        print(f"judged-relevant docs per topic: mean={merged['n_relevant_judged'].mean():.1f}  "
              f"median={merged['n_relevant_judged'].median():.1f}  "
              f"min={merged['n_relevant_judged'].min()}")

        corr_len_general = merged["word_count"].corr(merged["ndcg_general"])
        corr_len_biomed = merged["word_count"].corr(merged["ndcg_biomed"])
        corr_density_general = merged["n_relevant_judged"].corr(merged["ndcg_general"])
        corr_density_biomed = merged["n_relevant_judged"].corr(merged["ndcg_biomed"])

        print(f"\ncorr(topic length, nDCG)     general={corr_len_general:+.3f}   biomed={corr_len_biomed:+.3f}")
        print(f"corr(qrels density, nDCG)    general={corr_density_general:+.3f}   biomed={corr_density_biomed:+.3f}")

        print(f"\nworst 3 biomed topics (by nDCG):")
        worst = merged.nsmallest(3, "ndcg_biomed")
        for _, r in worst.iterrows():
            print(f"  topic {r['query_id']}: ndcg_biomed={r['ndcg_biomed']:.3f}  "
                  f"ndcg_general={r['ndcg_general']:.3f}  words={r['word_count']}  "
                  f"n_relevant_judged={r['n_relevant_judged']}")


if __name__ == "__main__":
    main()
