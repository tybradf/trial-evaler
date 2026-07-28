"""
Tests the hypothesis: does a longer exclusion-criteria list make the judge
more likely to wrongly exclude a truly-eligible patient (more surface area
= more chances to find something to hang an exclusion on)?

Compares exclusion-list length between:
  - false_exclude: ground_truth=eligible(2), predicted=excluded (the error)
  - true_include:  ground_truth=eligible(2), predicted=eligible (correct)
as the cleanest contrast -- same ground truth class, different outcome.

    python src/eval/check_exclusion_length_hypothesis.py
"""
import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
JUDGE_DIR = PROCESSED_DIR / "judge_results"


def main():
    frames = []
    for path in sorted(JUDGE_DIR.glob("results_*.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        df = pd.DataFrame(rows)
        df["combo"] = path.stem.replace("results_", "")
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    corpus = pd.read_parquet(PROCESSED_DIR / "corpus.parquet")
    corpus["excl_len"] = corpus["exclusion_criteria"].apply(len)
    corpus["incl_len"] = corpus["inclusion_criteria"].apply(len)

    merged = all_df.merge(corpus[["nct_id", "excl_len", "incl_len"]],
                           left_on="doc_id", right_on="nct_id", how="left")

    eligible_gt = merged[merged["ground_truth"] == 2]
    false_exclude = eligible_gt[eligible_gt["label"] == "excluded"]
    true_include = eligible_gt[eligible_gt["label"] == "eligible"]

    print(f"n false_exclude (gt=eligible, predicted=excluded): {len(false_exclude)}")
    print(f"n true_include  (gt=eligible, predicted=eligible):  {len(true_include)}")
    print()
    print(f"mean exclusion-list length -- false_exclude: {false_exclude['excl_len'].mean():.2f}")
    print(f"mean exclusion-list length -- true_include:  {true_include['excl_len'].mean():.2f}")
    print()
    print(f"median exclusion-list length -- false_exclude: {false_exclude['excl_len'].median():.1f}")
    print(f"median exclusion-list length -- true_include:  {true_include['excl_len'].median():.1f}")

    corr = eligible_gt["excl_len"].corr((eligible_gt["label"] == "excluded").astype(int))
    print(f"\ncorrelation(exclusion-list length, wrongly-excluded) on gt=eligible subset: {corr:+.3f}")


if __name__ == "__main__":
    main()
