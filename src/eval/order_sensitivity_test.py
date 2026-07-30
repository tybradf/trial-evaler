"""
Order-sensitivity test: a known failure mode in the LLM-as-judge
literature is that judgments can be sensitive to the order items are
presented in, independent of their actual content. For this task, that
would mean: does shuffling the order of a trial's inclusion/exclusion
criteria change the judge's verdict for the same patient and the same
underlying facts?

For a sample of pairs, calls the judge twice per pair -- once with
criteria in their original order, once with both lists independently
shuffled -- and reports the rate at which the label flips. A well-behaved
judge should show a low flip rate; the *content* didn't change, only the
order it was presented in.

    python src/eval/order_sensitivity_test.py --n_sample 40
"""
import argparse
import ast
import json
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judge"))
from judge_client import judge_pair

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "order_sensitivity_results.jsonl"


def parse_list_col(x):
    if isinstance(x, list):
        return x
    if pd.isna(x) or x == "":
        return []
    return ast.literal_eval(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_path", default=str(PROCESSED_DIR / "judge_test_sample_2022.csv"))
    ap.add_argument("--n_sample", type=int, default=40)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--strategy", default="zero_shot")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_csv(args.sample_path)
    df["inclusion_criteria"] = df["inclusion_criteria"].apply(parse_list_col)
    df["exclusion_criteria"] = df["exclusion_criteria"].apply(parse_list_col)

    # Only pairs with >=2 items in at least one list can actually be
    # shuffled into a different order -- skip single-item/empty lists,
    # shuffling them is a no-op and would just dilute the flip rate with
    # cases that were never actually tested.
    shufflable = df[(df["inclusion_criteria"].apply(len) >= 2) |
                     (df["exclusion_criteria"].apply(len) >= 2)]
    sample = shufflable.sample(n=min(args.n_sample, len(shufflable)), random_state=args.seed)
    print(f"Testing order sensitivity on {len(sample)} pairs "
          f"({args.model}/{args.strategy}, 2 calls each = {len(sample)*2} total calls)")

    rng = random.Random(args.seed)
    results = []
    flips = 0

    for row in tqdm(sample.itertuples(), total=len(sample)):
        original = judge_pair(
            patient_text=row.patient_text, trial_title=row.brief_title,
            inclusion=row.inclusion_criteria, exclusion=row.exclusion_criteria,
            model_key=args.model, strategy=args.strategy,
        )

        shuffled_inclusion = row.inclusion_criteria[:]
        shuffled_exclusion = row.exclusion_criteria[:]
        rng.shuffle(shuffled_inclusion)
        rng.shuffle(shuffled_exclusion)

        shuffled = judge_pair(
            patient_text=row.patient_text, trial_title=row.brief_title,
            inclusion=shuffled_inclusion, exclusion=shuffled_exclusion,
            model_key=args.model, strategy=args.strategy,
        )

        flipped = original["label"] != shuffled["label"]
        flips += flipped
        results.append({
            "query_id": row.query_id, "doc_id": row.doc_id,
            "original_label": original["label"], "shuffled_label": shuffled["label"],
            "flipped": flipped,
        })

    with open(OUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    rate = flips / len(sample) if len(sample) else float("nan")
    print(f"\nOrder-sensitivity flip rate: {flips}/{len(sample)} = {rate:.1%}")
    print(f"Results saved -> {OUT_PATH}")
    if flips:
        print("\nFlipped cases (verdict changed purely from criteria reordering, "
              "same underlying facts):")
        for r in results:
            if r["flipped"]:
                print(f"  query={r['query_id']} doc={r['doc_id']}: "
                      f"{r['original_label']} -> {r['shuffled_label']}")


if __name__ == "__main__":
    main()
