"""
Cost/quality frontier: combine actual $ cost (from logged token counts) with
the quality metrics from judge_metrics.py, per model x prompt-strategy combo.

Pricing as of 2026-07-28 (verify at platform.claude.com/docs/en/about-claude/pricing
if re-running after 2026-08-31, when Sonnet 5 moves off introductory pricing):
    haiku-4.5: $1.00 / $5.00 per million input/output tokens
    sonnet-5:  $2.00 / $10.00 per million input/output tokens (introductory,
               through 2026-08-31; standard $3.00/$15.00 after)

    python src/eval/cost_quality_frontier.py                          # dev (default)
    python src/eval/cost_quality_frontier.py \\
        --judge_dir data/processed/judge_results_test2022 \\
        --out_prefix test2022                                         # held-out test
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# $ per million tokens (input, output) -- dated, see module docstring
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
}


def compute_cost_per_call(row) -> float:
    price = PRICING[row["model"]]
    return (row["input_tokens"] / 1_000_000 * price["input"] +
            row["output_tokens"] / 1_000_000 * price["output"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_dir", default=str(PROCESSED_DIR / "judge_results"))
    ap.add_argument("--out_prefix", default="")
    args = ap.parse_args()

    judge_dir = Path(args.judge_dir)
    suffix = f"_{args.out_prefix}" if args.out_prefix else ""
    summary_path = PROCESSED_DIR / f"judge_eval_summary{suffix}.csv"
    out_path = PROCESSED_DIR / f"cost_quality_frontier{suffix}.csv"

    frames = []
    for path in sorted(judge_dir.glob("results_*.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        df = pd.DataFrame(rows)
        df["combo"] = path.stem.replace("results_", "")
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    all_df["cost_usd"] = all_df.apply(compute_cost_per_call, axis=1)

    quality = pd.read_csv(summary_path)

    cost_summary = all_df.groupby("combo").agg(
        n_calls=("cost_usd", "size"),
        mean_input_tokens=("input_tokens", "mean"),
        mean_output_tokens=("output_tokens", "mean"),
        total_cost_usd=("cost_usd", "sum"),
    ).reset_index()
    cost_summary["cost_per_1k_pairs_usd"] = (
        cost_summary["total_cost_usd"] / cost_summary["n_calls"] * 1000
    )

    frontier = quality.merge(cost_summary, on="combo")
    frontier = frontier.sort_values("cost_per_1k_pairs_usd")

    display_cols = ["combo", "cost_per_1k_pairs_usd", "accuracy_strict",
                     "mean_cost", "abstention_rate", "missed_exclusion_rate"]
    print(frontier[display_cols].rename(
        columns={"mean_cost": "mean_clinical_cost"}
    ).to_string(index=False))

    frontier.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    cheapest = frontier.iloc[0]
    most_expensive = frontier.iloc[-1]
    print(f"\nCheapest combo: {cheapest['combo']} "
          f"(${cheapest['cost_per_1k_pairs_usd']:.2f}/1k pairs, "
          f"accuracy={cheapest['accuracy_strict']:.3f}, "
          f"clinical_cost={cheapest['mean_cost']:.3f})")
    print(f"Most expensive combo: {most_expensive['combo']} "
          f"(${most_expensive['cost_per_1k_pairs_usd']:.2f}/1k pairs, "
          f"accuracy={most_expensive['accuracy_strict']:.3f}, "
          f"clinical_cost={most_expensive['mean_cost']:.3f})")


if __name__ == "__main__":
    main()
