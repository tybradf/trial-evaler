"""
Build a stratified sample of (patient, trial) pairs for the judge, from a
specified TREC year's qrels. Use --year 2021 (default) for dev/iteration
work, --year 2022 for the final held-out test -- run this on 2022 only
once the model/prompt/config is locked in from dev-set work, not before.

Scope: relevance in {1 (excluded), 2 (eligible)} only. Relevance=0 pairs are
"wrong disease area entirely" -- trivial for a judge and not the interesting
eligibility-reasoning task. Excludes trials flagged eligibility_needs_review
by default, to keep the primary accuracy sample clean of known parsing edge
cases (those get a separate stratified pass later for the error taxonomy).

    python src/judge/sample_pairs.py --year 2021 --n_per_class 60   # dev (default)
    python src/judge/sample_pairs.py --year 2022 --n_per_class 60   # held-out test
"""
import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2021, choices=[2021, 2022])
    ap.add_argument("--n_per_class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_needs_review", action="store_true",
                     help="include eligibility_needs_review trials (off by default)")
    args = ap.parse_args()

    out_name = "judge_dev_sample.csv" if args.year == 2021 else f"judge_test_sample_{args.year}.csv"
    out_path = PROCESSED_DIR / out_name

    qrels = pd.read_csv(RAW_DIR / f"qrels_{args.year}.csv")
    qrels = qrels[qrels["relevance"].isin([1, 2])]

    corpus = pd.read_parquet(PROCESSED_DIR / "corpus.parquet")
    if not args.include_needs_review:
        corpus = corpus[~corpus["eligibility_needs_review"]]

    topics = pd.read_csv(RAW_DIR / f"topics_{args.year}.csv")

    merged = qrels.merge(corpus, left_on="doc_id", right_on="nct_id", how="inner")
    merged = merged.merge(topics, on="query_id", how="inner")

    samples = []
    for rel in [1, 2]:
        subset = merged[merged["relevance"] == rel]
        n = min(args.n_per_class, len(subset))
        if n < args.n_per_class:
            print(f"WARNING: only {len(subset)} available for relevance={rel}, "
                  f"requested {args.n_per_class}")
        samples.append(subset.sample(n=n, random_state=args.seed))

    sample = pd.concat(samples).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    keep_cols = ["query_id", "text", "doc_id", "brief_title", "inclusion_criteria",
                 "exclusion_criteria", "relevance"]
    sample = sample[keep_cols].rename(columns={"text": "patient_text", "relevance": "ground_truth"})

    # IMPORTANT: corpus.parquet's list columns come back from pd.read_parquet
    # as numpy arrays, not native Python lists. str()'ing a numpy array of
    # strings omits the commas a real Python list would have, and reading
    # that back with ast.literal_eval() doesn't error -- it silently exploits
    # Python's adjacent-string-literal concatenation and merges every item
    # into one, with no space at the seam (confirmed: "...upper limb" +
    # "Subject with..." became "...upper limbSubject with..."). Explicit JSON
    # serialization has no such ambiguity and is what every reader of this
    # CSV (run_judge.py, data_prep.py, the bias-check scripts) now expects.
    import json
    sample["inclusion_criteria"] = sample["inclusion_criteria"].apply(lambda x: json.dumps(list(x)))
    sample["exclusion_criteria"] = sample["exclusion_criteria"].apply(lambda x: json.dumps(list(x)))

    sample.to_csv(out_path, index=False)

    print(f"[{args.year}] sample saved -> {out_path}")
    print(sample["ground_truth"].value_counts().rename({1: "excluded", 2: "eligible"}))


if __name__ == "__main__":
    main()
