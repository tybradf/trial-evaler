"""
Run locally (needs your real data/processed/ outputs) to bundle everything
the Flask app needs into a single JSON file. The deployed app reads only
this file -- no pandas/pyarrow/torch dependency in production.

    python app/data_prep.py

Re-run this any time the underlying eval data changes; the app always
reads app/data/dashboard_data.json, never the raw CSVs/parquet directly.
"""
import ast
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_PATH = Path(__file__).resolve().parent / "data" / "dashboard_data.json"

sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))
from judge_metrics import confusion_table, LABEL_NAME  # reuse, don't duplicate


def parse_list_col(x):
    if isinstance(x, list):
        return x
    if pd.isna(x) or x == "":
        return []
    return ast.literal_eval(x)


def load_judge_results(judge_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(judge_dir.glob("results_*.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        df = pd.DataFrame(rows)
        df["combo"] = path.stem.replace("results_", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_confusion_matrices(all_df: pd.DataFrame) -> dict:
    out = {}
    for combo, group in all_df.groupby("combo"):
        ct = confusion_table(group)
        out[combo] = {row: {col: int(ct.loc[row, col]) for col in ct.columns}
                      for row in ct.index}
    return out


def build_explorer_pairs(sample_path: Path, judge_df: pd.DataFrame) -> list:
    sample = pd.read_csv(sample_path)
    sample["inclusion_criteria"] = sample["inclusion_criteria"].apply(parse_list_col)
    sample["exclusion_criteria"] = sample["exclusion_criteria"].apply(parse_list_col)

    pairs = []
    for _, row in sample.iterrows():
        verdicts = {}
        subset = judge_df[(judge_df["query_id"] == row["query_id"]) &
                           (judge_df["doc_id"] == row["doc_id"])]
        for _, jrow in subset.iterrows():
            verdicts[jrow["combo"]] = {
                "label": jrow["label"],
                "cited_criterion": jrow["cited_criterion"],
                "rationale": jrow["rationale"],
            }
        pairs.append({
            "query_id": int(row["query_id"]),
            "doc_id": row["doc_id"],
            "patient_text": row["patient_text"],
            "trial_title": row["brief_title"],
            "inclusion_criteria": row["inclusion_criteria"],
            "exclusion_criteria": row["exclusion_criteria"],
            "ground_truth": LABEL_NAME[int(row["ground_truth"])],
            "verdicts": verdicts,
        })
    return pairs


def df_to_records(path: Path) -> list:
    return json.loads(pd.read_csv(path).to_json(orient="records"))


def main():
    data = {}

    data["retrieval_metrics"] = df_to_records(PROCESSED_DIR / "retrieval_metrics.csv")
    data["judge_dev_summary"] = df_to_records(PROCESSED_DIR / "judge_eval_summary.csv")
    data["judge_test_summary"] = df_to_records(PROCESSED_DIR / "judge_eval_summary_test2022.csv")
    data["cost_quality_dev"] = df_to_records(PROCESSED_DIR / "cost_quality_frontier.csv")
    data["cost_quality_test"] = df_to_records(PROCESSED_DIR / "cost_quality_frontier_test2022.csv")

    dev_results = load_judge_results(PROCESSED_DIR / "judge_results")
    test_results = load_judge_results(PROCESSED_DIR / "judge_results_test2022")
    data["confusion_matrices_dev"] = build_confusion_matrices(dev_results)
    data["confusion_matrices_test"] = build_confusion_matrices(test_results)

    data["explorer_pairs"] = build_explorer_pairs(
        PROCESSED_DIR / "judge_test_sample_2022.csv", test_results
    )

    test_summary = {r["combo"]: r for r in data["judge_test_summary"]}
    sonnet = test_summary.get("sonnet_zero_shot", {})
    data["headline"] = {
        "n_trials": 48713,
        "n_topics_2021": 75,
        "n_topics_2022": 50,
        "sonnet_missed_exclusion_rate": sonnet.get("missed_exclusion_rate"),
        "sonnet_accuracy_committed": sonnet.get("accuracy_committed"),
        "sonnet_abstention_rate": sonnet.get("abstention_rate"),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(data, indent=2, default=str))
    print(f"Bundled -> {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"{len(data['explorer_pairs'])} explorer pairs, "
          f"{len(data['confusion_matrices_test'])} test combos with confusion matrices.")


if __name__ == "__main__":
    main()
