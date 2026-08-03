"""
Self-consistency test: does the judge give the same answer when asked the
identical question multiple times? This is a distinct eval dimension from
accuracy -- a judge could be perfectly calibrated on average but unstable
on individual cases, which matters operationally (does the same
patient-trial pair get a different answer depending on when you ask?).

For a sample of pairs, calls the judge k times per pair on the exact same
input (same criteria order, nothing varied) and reports what fraction of
pairs got a unanimous verdict across all k runs.

    python src/eval/self_consistency_test.py --n_sample 40 --k 3
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judge"))
from judge_client import judge_pair

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "self_consistency_results.jsonl"


def parse_list_col(x):
    """See src/judge/sample_pairs.py's comment: criteria columns are
    JSON-encoded on write, must be json.loads'd on read."""
    if isinstance(x, list):
        return x
    if pd.isna(x) or x == "":
        return []
    return json.loads(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_path", default=str(PROCESSED_DIR / "judge_test_sample_2022.csv"))
    ap.add_argument("--n_sample", type=int, default=40)
    ap.add_argument("--k", type=int, default=3, help="repeats per pair")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--strategy", default="zero_shot")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    df = pd.read_csv(args.sample_path)
    df["inclusion_criteria"] = df["inclusion_criteria"].apply(parse_list_col)
    df["exclusion_criteria"] = df["exclusion_criteria"].apply(parse_list_col)
    sample = df.sample(n=min(args.n_sample, len(df)), random_state=args.seed)

    print(f"Testing self-consistency on {len(sample)} pairs, {args.k} repeats each "
          f"({args.model}/{args.strategy}, {len(sample)*args.k} total calls)")

    results = []
    unanimous_count = 0

    for row in tqdm(sample.itertuples(), total=len(sample)):
        labels = []
        for _ in range(args.k):
            r = judge_pair(
                patient_text=row.patient_text, trial_title=row.brief_title,
                inclusion=row.inclusion_criteria, exclusion=row.exclusion_criteria,
                model_key=args.model, strategy=args.strategy,
            )
            labels.append(r["label"])

        counts = Counter(labels)
        majority_label, majority_count = counts.most_common(1)[0]
        unanimous = majority_count == args.k
        unanimous_count += unanimous

        results.append({
            "query_id": row.query_id, "doc_id": row.doc_id,
            "labels": labels, "unanimous": unanimous,
            "majority_label": majority_label, "majority_count": majority_count,
        })

    with open(OUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    rate = unanimous_count / len(sample) if len(sample) else float("nan")
    print(f"\nSelf-consistency rate: {unanimous_count}/{len(sample)} pairs "
          f"unanimous across {args.k} runs = {rate:.1%}")
    print(f"Results saved -> {OUT_PATH}")

    inconsistent = [r for r in results if not r["unanimous"]]
    if inconsistent:
        print(f"\n{len(inconsistent)} inconsistent pairs (verdict varied across "
              f"identical repeated calls):")
        for r in inconsistent:
            print(f"  query={r['query_id']} doc={r['doc_id']}: labels={r['labels']}")


if __name__ == "__main__":
    main()
