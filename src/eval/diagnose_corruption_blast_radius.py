"""
Quantifies the blast radius of the numpy-array/ast.literal_eval CSV
corruption bug (see src/judge/sample_pairs.py's comment for the full
mechanism) on an already-drawn sample, WITHOUT any API calls. Compares
each pair's stored criteria against a fresh, correct build from
corpus.parquet and reports how many pairs actually differ.

    python src/eval/diagnose_corruption_blast_radius.py \\
        --sample_path data/processed/judge_test_sample_2022.csv
"""
import argparse
import ast
import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def parse_maybe_corrupted(x):
    """Tries JSON first (the fixed format); falls back to the old
    ast.literal_eval path so this diagnostic can read samples drawn
    before the fix, which is exactly what we need to check here."""
    if isinstance(x, list):
        return x
    if pd.isna(x) or x == "":
        return []
    try:
        return json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_path", default=str(PROCESSED_DIR / "judge_test_sample_2022.csv"))
    args = ap.parse_args()

    sample = pd.read_csv(args.sample_path)
    sample["inclusion_criteria"] = sample["inclusion_criteria"].apply(parse_maybe_corrupted)
    sample["exclusion_criteria"] = sample["exclusion_criteria"].apply(parse_maybe_corrupted)

    corpus = pd.read_parquet(PROCESSED_DIR / "corpus.parquet").set_index("nct_id")

    affected_rows = []
    for _, row in sample.iterrows():
        if row["doc_id"] not in corpus.index:
            continue
        true_incl = list(corpus.loc[row["doc_id"], "inclusion_criteria"])
        true_excl = list(corpus.loc[row["doc_id"], "exclusion_criteria"])

        incl_mismatch = list(row["inclusion_criteria"]) != true_incl
        excl_mismatch = list(row["exclusion_criteria"]) != true_excl

        if incl_mismatch or excl_mismatch:
            affected_rows.append({
                "query_id": row["query_id"], "doc_id": row["doc_id"],
                "stored_incl_count": len(row["inclusion_criteria"]),
                "true_incl_count": len(true_incl),
                "stored_excl_count": len(row["exclusion_criteria"]),
                "true_excl_count": len(true_excl),
            })

    n_total = len(sample)
    n_affected = len(affected_rows)
    print(f"Total pairs in sample: {n_total}")
    print(f"Pairs with corrupted criteria (stored != corpus.parquet ground truth): "
          f"{n_affected} ({n_affected/n_total:.1%})")

    if affected_rows:
        affected_df = pd.DataFrame(affected_rows)
        out_path = Path(args.sample_path).parent / "corruption_blast_radius.csv"
        affected_df.to_csv(out_path, index=False)
        print(f"Details saved -> {out_path}")
        print(f"\nExample affected rows:")
        print(affected_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
