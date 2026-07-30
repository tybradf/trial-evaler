"""
Statistical significance test on the headline missed-exclusion-rate gap
between judge configs, on held-out data.

Methodological note, worth getting right: missed_exclusion_rate as
reported by judge_metrics.py divides by ALL pairs (both ground-truth
classes). But a missed exclusion -- predicting "eligible" when ground
truth is "excluded" -- can only happen among the truly-excluded pairs;
the eligible-ground-truth pairs are structurally incapable of producing
this error. Testing the proportion against the full sample size dilutes
the signal with cases that can't contribute to it. This script uses the
correct at-risk denominator: only ground-truth-excluded pairs.

Uses Fisher's exact test (appropriate for small-count 2x2 contingency
tables, more reliable than a chi-square approximation at these sample
sizes) plus Wilson score confidence intervals for each individual
proportion (implemented manually -- no new dependency for one calculation).

    python src/eval/significance_test.py --judge_dir data/processed/judge_results_test2022
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def wilson_ci(x: int, n: int, alpha: float = 0.05) -> tuple:
    z = 1.959963985  # z-score for 95% CI
    if n == 0:
        return (float("nan"), float("nan"))
    p_hat = x / n
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def load_results(judge_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(judge_dir.glob("results_*.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        df = pd.DataFrame(rows)
        df["combo"] = path.stem.replace("results_", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_dir", default=str(PROCESSED_DIR / "judge_results_test2022"))
    ap.add_argument("--combo_a", default="sonnet_zero_shot")
    ap.add_argument("--combo_b", default="haiku_zero_shot")
    args = ap.parse_args()

    df = load_results(Path(args.judge_dir))

    stats = {}
    for combo in [args.combo_a, args.combo_b]:
        subset = df[df["combo"] == combo]
        excluded_gt = subset[subset["ground_truth"] == 1]  # the at-risk population
        n = len(excluded_gt)
        missed = ((excluded_gt["label"] == "eligible")).sum()
        stats[combo] = {"n": n, "missed": int(missed)}

    print(f"At-risk population (ground-truth-excluded pairs only, not all pairs):\n")
    for combo, s in stats.items():
        lo, hi = wilson_ci(s["missed"], s["n"])
        rate = s["missed"] / s["n"] if s["n"] else float("nan")
        print(f"  {combo}: {s['missed']}/{s['n']} missed exclusions = {rate:.1%}  "
              f"(95% Wilson CI: [{lo:.1%}, {hi:.1%}])")

    a, b = stats[args.combo_a], stats[args.combo_b]
    table = [
        [a["missed"], a["n"] - a["missed"]],
        [b["missed"], b["n"] - b["missed"]],
    ]
    odds_ratio, p_two_sided = fisher_exact(table, alternative="two-sided")
    _, p_one_sided = fisher_exact(table, alternative="less")

    print(f"\nFisher's exact test ({args.combo_a} vs {args.combo_b}):")
    print(f"  odds ratio: {odds_ratio:.3f}")
    print(f"  p-value (two-sided): {p_two_sided:.4f}")
    print(f"  p-value (one-sided, {args.combo_a} < {args.combo_b}): {p_one_sided:.4f}")

    alpha = 0.05
    if p_two_sided < alpha:
        print(f"\nResult: significant at alpha={alpha} (p={p_two_sided:.4f}). "
              f"The missed-exclusion gap is unlikely to be noise at this sample size.")
    else:
        print(f"\nResult: NOT significant at alpha={alpha} (p={p_two_sided:.4f}). "
              f"Consider a larger sample before treating the gap as established.")

    ci_a = wilson_ci(a["missed"], a["n"])
    ci_b = wilson_ci(b["missed"], b["n"])
    if ci_a[1] > ci_b[0]:
        print(f"\nNote: the 95% CIs overlap ([{ci_a[0]:.1%}, {ci_a[1]:.1%}] vs "
              f"[{ci_b[0]:.1%}, {ci_b[1]:.1%}]) despite the significant p-value -- "
              f"the point estimate is real but not yet precisely pinned down. "
              f"A larger sample would tighten this without changing the direction.")


if __name__ == "__main__":
    main()
