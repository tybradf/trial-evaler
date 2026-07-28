"""
Print the confident_disagreement pairs (all/most combos confidently agreed
on the same wrong label) with full patient/trial context -- these are the
most informative of the ambiguous bucket and worth reading individually,
unlike the consensus_hedge/mixed cases which are already explained by
vignette insufficiency.

    python src/eval/read_confident_disagreements.py
"""
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

LABEL_NAME = {1: "excluded", 2: "eligible"}


def main():
    ambiguous = pd.read_csv(PROCESSED_DIR / "judge_candidate_ground_truth_ambiguity.csv")
    confident = ambiguous[ambiguous["pattern"] == "confident_disagreement"]

    topics = pd.read_csv(RAW_DIR / "topics_2021.csv").set_index("query_id")
    corpus = pd.read_parquet(PROCESSED_DIR / "corpus.parquet").set_index("nct_id")

    print(f"{len(confident)} confident_disagreement pairs\n")
    for _, row in confident.iterrows():
        qid, nct = row["query_id"], row["doc_id"]
        gt_name = LABEL_NAME[row["ground_truth"]]
        print(f"{'=' * 70}")
        print(f"query {qid} / {nct}  --  ground truth: {gt_name}  "
              f"--  models confidently said: {row['consensus_wrong_label']}")
        print(f"{'=' * 70}")
        print("\nPATIENT:")
        print(topics.loc[qid, "text"])
        meta = corpus.loc[nct]
        print(f"\nTRIAL: {meta['brief_title']}")
        print(f"\nINCLUSION: {meta['inclusion_criteria']}")
        print(f"\nEXCLUSION: {meta['exclusion_criteria']}")
        print()


if __name__ == "__main__":
    main()
