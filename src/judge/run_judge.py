"""
Run the judge across a sample, for the specified (model, prompt strategy)
combinations, and save full transcripts + structured outputs + token counts.

Default behavior (no args) matches the original Day 3 dev run: all models x
all strategies against the 2021 dev sample. For the 2022 held-out test with
a single locked-in config:

    python src/judge/sample_pairs.py --year 2022 --n_per_class 60
    python src/judge/run_judge.py \\
        --sample_path data/processed/judge_test_sample_2022.csv \\
        --out_dir data/processed/judge_results_test2022 \\
        --models sonnet --strategies zero_shot

Cost estimate before running: n_pairs x n_models x n_strategies calls, at a
few hundred tokens each. Check the sample file's row count first if you
changed --n_per_class.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from judge_client import judge_pair, MODELS

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def parse_list_col(x):
    """inclusion_criteria/exclusion_criteria round-tripped through CSV as
    JSON-encoded lists (sample_pairs.py explicitly json.dumps() these --
    see that file's comment on why ast.literal_eval was unsafe here: it
    silently mis-parsed numpy-array-sourced cells via Python's
    adjacent-string-literal concatenation, merging multiple criteria into
    one with no space at the seam). Parse with json.loads to match."""
    if isinstance(x, list):
        return x
    if pd.isna(x) or x == "":
        return []
    return json.loads(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_path", default=str(PROCESSED_DIR / "judge_dev_sample.csv"))
    ap.add_argument("--out_dir", default=str(PROCESSED_DIR / "judge_results"))
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                     choices=list(MODELS.keys()), help="e.g. --models sonnet")
    ap.add_argument("--strategies", nargs="+", default=["zero_shot", "few_shot"],
                     choices=["zero_shot", "few_shot"])
    args = ap.parse_args()

    sample_path = Path(args.sample_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(sample_path)
    sample["inclusion_criteria"] = sample["inclusion_criteria"].apply(parse_list_col)
    sample["exclusion_criteria"] = sample["exclusion_criteria"].apply(parse_list_col)

    total_calls = len(sample) * len(args.models) * len(args.strategies)
    print(f"Loaded {len(sample)} pairs from {sample_path.name}. Running "
          f"{len(args.models)} model(s) x {len(args.strategies)} strateg(y/ies) "
          f"= {total_calls} total calls -> {out_dir}")

    for model_key in args.models:
        for strategy in args.strategies:
            out_path = out_dir / f"results_{model_key}_{strategy}.jsonl"
            if out_path.exists():
                print(f"SKIP (already exists): {out_path.name} -- delete it to re-run")
                continue

            print(f"\n=== {model_key} / {strategy} ===")
            results = []
            errors = 0
            for row in tqdm(sample.itertuples(), total=len(sample)):
                try:
                    result = judge_pair(
                        patient_text=row.patient_text,
                        trial_title=row.brief_title,
                        inclusion=row.inclusion_criteria,
                        exclusion=row.exclusion_criteria,
                        model_key=model_key,
                        strategy=strategy,
                    )
                    result.update({
                        "query_id": row.query_id,
                        "doc_id": row.doc_id,
                        "ground_truth": row.ground_truth,  # 1=excluded, 2=eligible
                    })
                    results.append(result)
                except Exception as e:
                    errors += 1
                    print(f"\nFAILED on query={row.query_id} doc={row.doc_id}: {e}")

            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

            print(f"{model_key}/{strategy}: {len(results)} succeeded, {errors} failed "
                  f"-> {out_path}")


if __name__ == "__main__":
    main()
