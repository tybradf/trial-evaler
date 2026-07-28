"""
The centerpiece eval: score every (model, prompt strategy) combo's judge
output against the physician qrels ground truth.

Ground truth is 2-class (1=excluded, 2=eligible). The judge predicts 3
classes (excluded / eligible / insufficient_information). Metrics:

  - Per-class precision/recall/F1 for "excluded" and "eligible" -- an
    abstention counts against recall (a real miss) but not against the
    opposite class's precision (it didn't wrongly assert anything).
  - abstention_rate -- how often the judge punts.
  - accuracy_strict -- correct / total, abstentions counted as wrong.
  - accuracy_committed -- correct / (total - abstentions), i.e. accuracy
    only among answers the judge actually committed to.
  - mean_cost -- a clinically-weighted score reflecting that a missed
    exclusion (ground truth=excluded, predicted=eligible) is worse than an
    over-exclusion (ground truth=eligible, predicted=excluded), which is
    worse than a hedge, which is worse than being correct. Weights are a
    documented judgment call, not a standard metric -- see COST below.

    python src/eval/judge_metrics.py                                    # dev (default)
    python src/eval/judge_metrics.py \\
        --judge_dir data/processed/judge_results_test2022 \\
        --out_prefix test2022                                           # held-out test

Note: the ground-truth-ambiguity cross-combo comparison is most meaningful
with multiple (model, strategy) combos to compare agreement across -- with
a single combo (e.g. the held-out test run), every wrong answer trivially
"disagrees" at 100%, so that output is less informative there. Still
computed for consistency, just don't over-read it with n=1 combo.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# ground_truth (1=excluded, 2=eligible), predicted_label -> cost
# 5.0 : missed exclusion -- the judge cleared a patient the physician excluded
# 1.0 : over-exclusion -- the judge excluded a patient the physician cleared
# 0.5 : hedge -- safe but unhelpful; needs a human to resolve either way
# 0.0 : correct
COST = {
    (1, "eligible"): 5.0,
    (2, "excluded"): 1.0,
    (1, "excluded"): 0.0,
    (2, "eligible"): 0.0,
    (1, "insufficient_information"): 0.5,
    (2, "insufficient_information"): 0.5,
}

LABEL_NAME = {1: "excluded", 2: "eligible"}


def load_all_results(judge_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(judge_dir.glob("results_*.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        df = pd.DataFrame(rows)
        df["combo"] = path.stem.replace("results_", "")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No results_*.jsonl files found in {judge_dir} -- run run_judge.py first.")
    return pd.concat(frames, ignore_index=True)


def compute_metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    df = df.copy()
    df["cost"] = df.apply(lambda r: COST[(r["ground_truth"], r["label"])], axis=1)
    df["correct"] = (
        ((df["ground_truth"] == 1) & (df["label"] == "excluded")) |
        ((df["ground_truth"] == 2) & (df["label"] == "eligible"))
    )
    abstained = (df["label"] == "insufficient_information").sum()

    metrics = {
        "n": n,
        "accuracy_strict": df["correct"].mean(),
        "accuracy_committed": df.loc[df["label"] != "insufficient_information", "correct"].mean(),
        "abstention_rate": abstained / n,
        "mean_cost": df["cost"].mean(),
        "missed_exclusion_rate": ((df["ground_truth"] == 1) & (df["label"] == "eligible")).mean(),
    }

    for gt_class, gt_name in LABEL_NAME.items():
        pred_name = gt_name
        tp = ((df["ground_truth"] == gt_class) & (df["label"] == pred_name)).sum()
        fn = ((df["ground_truth"] == gt_class) & (df["label"] != pred_name)).sum()
        fp = ((df["ground_truth"] != gt_class) & (df["label"] == pred_name)).sum()
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
        metrics[f"recall_{gt_name}"] = recall
        metrics[f"precision_{gt_name}"] = precision
        metrics[f"f1_{gt_name}"] = f1

    return metrics


def confusion_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ground_truth_name"] = df["ground_truth"].map(LABEL_NAME)
    ct = pd.crosstab(df["ground_truth_name"], df["label"])
    for col in ["excluded", "eligible", "insufficient_information"]:
        if col not in ct.columns:
            ct[col] = 0
    return ct[["excluded", "eligible", "insufficient_information"]]


def flag_ground_truth_ambiguity(all_df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """A pair is a candidate ground-truth-ambiguity case if at least
    `threshold` fraction of (model, strategy) combos disagree with ground
    truth on it. Splits into two sub-patterns that mean very different
    things:
      - consensus_hedge: all/most disagreeing combos said
        insufficient_information -- not really "disagreement," more likely
        the vignette genuinely lacks a detail the physician's chart had.
      - confident_disagreement: all/most disagreeing combos committed to
        the SAME wrong label -- a stronger, more surprising signal, worth
        individual scrutiny."""
    all_df = all_df.copy()
    all_df["correct"] = (
        ((all_df["ground_truth"] == 1) & (all_df["label"] == "excluded")) |
        ((all_df["ground_truth"] == 2) & (all_df["label"] == "eligible"))
    )

    def summarize(g):
        wrong = g[~g["correct"]]
        n_combos = len(g)
        n_disagree = len(wrong)
        n_hedge = (wrong["label"] == "insufficient_information").sum()
        n_confident_wrong = n_disagree - n_hedge
        # what did the confidently-wrong ones agree on, if anything?
        confident_labels = wrong.loc[wrong["label"] != "insufficient_information", "label"]
        consensus_wrong_label = confident_labels.mode().iloc[0] if len(confident_labels) else None
        return pd.Series({
            "n_combos": n_combos,
            "n_disagree": n_disagree,
            "disagree_rate": n_disagree / n_combos,
            "n_hedge": n_hedge,
            "n_confident_wrong": n_confident_wrong,
            "consensus_wrong_label": consensus_wrong_label,
            "ground_truth": g["ground_truth"].iloc[0],
        })

    grouped = all_df.groupby(["query_id", "doc_id"]).apply(summarize).reset_index()
    flagged = grouped[grouped["disagree_rate"] >= threshold].copy()

    flagged["pattern"] = flagged.apply(
        lambda r: "consensus_hedge" if r["n_confident_wrong"] == 0
        else ("confident_disagreement" if r["n_hedge"] == 0 else "mixed"), axis=1
    )
    return flagged.sort_values("disagree_rate", ascending=False)


def extract_errors_for_taxonomy(all_df: pd.DataFrame, ambiguous_pairs: pd.DataFrame) -> pd.DataFrame:
    """Every wrong (non-abstain, non-ambiguous-flagged) prediction, ready
    for manual error-category tagging. Leaves an empty error_category
    column -- deliberately not auto-classified; this is where clinical
    judgment is the actual value-add, not something to fake with keyword
    heuristics."""
    all_df = all_df.copy()
    all_df["correct"] = (
        ((all_df["ground_truth"] == 1) & (all_df["label"] == "excluded")) |
        ((all_df["ground_truth"] == 2) & (all_df["label"] == "eligible"))
    )
    wrong = all_df[
        (~all_df["correct"]) & (all_df["label"] != "insufficient_information")
    ].copy()

    ambiguous_keys = set(zip(ambiguous_pairs["query_id"], ambiguous_pairs["doc_id"]))
    wrong["is_candidate_ground_truth_ambiguity"] = wrong.apply(
        lambda r: (r["query_id"], r["doc_id"]) in ambiguous_keys, axis=1
    )

    wrong["error_category"] = ""  # fill in by hand: negation / temporal / numeric_threshold /
    # logical_scoping / hallucinated_criterion / missed_exclusion / other
    cols = ["combo", "query_id", "doc_id", "ground_truth", "label", "cited_criterion",
            "rationale", "is_candidate_ground_truth_ambiguity", "error_category"]
    return wrong[cols].sort_values(["is_candidate_ground_truth_ambiguity", "combo"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_dir", default=str(PROCESSED_DIR / "judge_results"))
    ap.add_argument("--out_prefix", default="", help="e.g. 'test2022' -> judge_eval_summary_test2022.csv")
    args = ap.parse_args()

    judge_dir = Path(args.judge_dir)
    suffix = f"_{args.out_prefix}" if args.out_prefix else ""
    out_summary = PROCESSED_DIR / f"judge_eval_summary{suffix}.csv"
    out_errors = PROCESSED_DIR / f"judge_errors_for_taxonomy{suffix}.csv"
    out_ambiguous = PROCESSED_DIR / f"judge_candidate_ground_truth_ambiguity{suffix}.csv"

    all_df = load_all_results(judge_dir)
    print(f"Loaded {len(all_df)} total judgments across {all_df['combo'].nunique()} combos.\n")

    summary_rows = []
    for combo, group in all_df.groupby("combo"):
        m = compute_metrics(group)
        m["combo"] = combo
        summary_rows.append(m)

        print(f"{'=' * 60}\n{combo}  (n={m['n']})\n{'=' * 60}")
        print(f"accuracy_strict:      {m['accuracy_strict']:.3f}")
        print(f"accuracy_committed:   {m['accuracy_committed']:.3f}")
        print(f"abstention_rate:      {m['abstention_rate']:.3f}")
        print(f"mean_cost:            {m['mean_cost']:.3f}")
        print(f"missed_exclusion_rate:{m['missed_exclusion_rate']:.3f}  (the dangerous error)")
        print(f"excluded  -- precision={m['precision_excluded']:.3f} recall={m['recall_excluded']:.3f} f1={m['f1_excluded']:.3f}")
        print(f"eligible  -- precision={m['precision_eligible']:.3f} recall={m['recall_eligible']:.3f} f1={m['f1_eligible']:.3f}")
        print("\nconfusion matrix (rows=ground truth, cols=predicted):")
        print(confusion_table(group).to_string())
        print()

    summary_df = pd.DataFrame(summary_rows).set_index("combo")
    summary_df.to_csv(out_summary)
    print(f"Summary saved -> {out_summary}")

    ambiguous = flag_ground_truth_ambiguity(all_df)
    ambiguous.to_csv(out_ambiguous, index=False)
    print(f"\n{len(ambiguous)} candidate ground-truth-ambiguity pairs (>=75% of combos disagree) "
          f"-> {out_ambiguous}")
    if all_df["combo"].nunique() > 1:
        print("Pattern breakdown:")
        print(ambiguous["pattern"].value_counts().to_string())
    else:
        print("(single combo -- ambiguity flag is just 'wrong answer', not a cross-model signal here)")

    errors = extract_errors_for_taxonomy(all_df, ambiguous)
    errors.to_csv(out_errors, index=False)
    n_flagged = errors["is_candidate_ground_truth_ambiguity"].sum()
    print(f"{len(errors)} total non-abstain errors ({n_flagged} flagged as candidate "
          f"ground-truth ambiguity, {len(errors) - n_flagged} likely real model errors) "
          f"-> {out_errors}")
    print("Fill in the empty error_category column by hand for the real-error rows.")


if __name__ == "__main__":
    main()
