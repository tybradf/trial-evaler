"""
Cheap self-consistency check: does the model's own rationale text contradict
its structured label? (e.g. rationale says "indeterminate"/"cannot be
determined" but label is "excluded" or "eligible", not
"insufficient_information"). This is a distinct failure mode from being
factually wrong -- it's the model not being coherent with itself.

    python src/judge/check_label_consistency.py
"""
import json
from pathlib import Path

INCONSISTENCY_PHRASES = [
    "indeterminate", "cannot be determined", "cannot be definitively",
    "unclear whether", "not enough information to", "insufficient to",
]

JUDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "judge_results"


def main():
    for path in sorted(JUDGE_DIR.glob("results_*.jsonl")):
        flagged = 0
        total = 0
        for line in path.open():
            r = json.loads(line)
            total += 1
            if r["label"] != "insufficient_information":
                text = r["rationale"].lower()
                if any(p in text for p in INCONSISTENCY_PHRASES):
                    flagged += 1
                    print(f"[{path.name}] query={r['query_id']} doc={r['doc_id']} "
                          f"label={r['label']} -- rationale hedges: \"{r['rationale'][:150]}...\"")
        print(f"{path.name}: {flagged}/{total} possibly label/rationale-inconsistent\n")


if __name__ == "__main__":
    main()
