"""
Cost/quality frontier chart: $/1k pairs (x) vs clinically-weighted mean
cost (y, lower=better), marker size = missed_exclusion_rate so the
dangerous-error rate is visible at a glance, not just in a tooltip.

    python src/eval/plot_cost_quality_frontier.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "cost_quality_frontier.png"

LABELS = {
    "haiku_zero_shot": "Haiku / zero-shot",
    "haiku_few_shot": "Haiku / few-shot",
    "sonnet_zero_shot": "Sonnet / zero-shot",
    "sonnet_few_shot": "Sonnet / few-shot",
}
COLORS = {
    "haiku_zero_shot": "#4C72B0",
    "haiku_few_shot": "#64B5CD",
    "sonnet_zero_shot": "#C44E52",
    "sonnet_few_shot": "#DD8452",
}


def main():
    df = pd.read_csv(PROCESSED_DIR / "cost_quality_frontier.csv")

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in df.iterrows():
        combo = row["combo"]
        size = 300 + row["missed_exclusion_rate"] * 4000  # visually separate 0% from ~5%
        ax.scatter(row["cost_per_1k_pairs_usd"], row["mean_cost"],
                   s=size, color=COLORS.get(combo, "gray"), alpha=0.75,
                   edgecolors="black", linewidths=1, zorder=3)
        ax.annotate(
            f"{LABELS.get(combo, combo)}\nmissed excl.={row['missed_exclusion_rate']:.0%}",
            (row["cost_per_1k_pairs_usd"], row["mean_cost"]),
            textcoords="offset points", xytext=(12, 8), fontsize=9,
        )

    ax.set_xlabel("Cost per 1,000 patient-trial pairs ($)", fontsize=11)
    ax.set_ylabel("Mean clinically-weighted cost (lower = better)", fontsize=11)
    ax.set_title("Judge cost/quality frontier\n(marker size = missed-exclusion rate, the dangerous error)",
                 fontsize=12)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Chart saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
