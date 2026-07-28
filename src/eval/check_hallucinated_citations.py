"""
Screen every "excluded" verdict for a candidate hallucination: does the
cited_criterion share ANY meaningful content word with the patient
vignette it was judged against? If not, the model may have asserted a
disqualifying fact not actually present in the source text -- confirmed
real for query 33/NCT01223885 (cited "Pregnant women" as the exclusion
reason; the vignette never mentions pregnancy status at all).

This is a heuristic triage tool, not a verdict -- word-overlap can miss
legitimate paraphrases (a real match phrased differently) and won't catch
every hallucination (a fabricated fact that happens to share a word with
the text). Flagged rows need a human read, same as everything else in this
project's error taxonomy. It's meant to surface candidates efficiently
across 480 rows, not replace judgment.

    python src/eval/check_hallucinated_citations.py
"""
import json
import re
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
JUDGE_DIR = PROCESSED_DIR / "judge_results"
OUT_PATH = PROCESSED_DIR / "candidate_hallucinated_citations.csv"

STOPWORDS = {
    "the", "a", "an", "of", "or", "and", "with", "without", "on", "in", "to", "for",
    "is", "are", "patient", "patients", "women", "men", "history", "not", "no", "have",
    "has", "had", "requiring", "required", "due", "any", "other", "this", "that", "from",
    "at", "by", "as", "was", "were", "will", "be", "than", "less", "more", "than",
}


def is_threshold_criterion(text: str) -> bool:
    """True for comparison/threshold-style criteria (age cutoffs, numeric
    ranges, lab-value comparisons) where NON-overlap between the citation
    and the vignette is expected, correct reasoning -- the citation states
    the trial's threshold, which is supposed to differ from the patient's
    actual value; that's what makes it a valid exclusion. These are
    excluded from the hallucination screen entirely, since flagging them
    produces near-100% false positives (confirmed: every one of the first
    64 flagged rows in the initial run was an age/threshold citation, and
    every single one was correctly reasoned)."""
    if re.search(r"[<>\u2264\u2265]", text):
        return True
    if re.search(r"\d+\s*-\s*\d+", text):
        return True
    if re.search(r"\bage\b", text.lower()) and re.search(r"\d", text):
        return True
    return False


def content_words(text: str) -> set:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def main():
    frames = []
    for path in sorted(JUDGE_DIR.glob("results_*.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        df = pd.DataFrame(rows)
        df["combo"] = path.stem.replace("results_", "")
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    excluded_only = all_df[all_df["label"] == "excluded"].copy()
    excluded_only = excluded_only[~excluded_only["cited_criterion"].apply(is_threshold_criterion)]

    topics_2021 = pd.read_csv(RAW_DIR / "topics_2021.csv").set_index("query_id")["text"]
    excluded_only["patient_text"] = excluded_only["query_id"].map(topics_2021)

    def overlap_ratio(row):
        cited = content_words(row["cited_criterion"])
        patient = content_words(row["patient_text"])
        if not cited:
            return None
        return len(cited & patient) / len(cited)

    excluded_only["overlap_ratio"] = excluded_only.apply(overlap_ratio, axis=1)
    candidates = excluded_only[excluded_only["overlap_ratio"] == 0.0].copy()

    cols = ["combo", "query_id", "doc_id", "ground_truth", "cited_criterion", "rationale", "patient_text"]
    candidates = candidates[cols].sort_values(["query_id", "doc_id"])
    candidates.to_csv(OUT_PATH, index=False)

    print(f"{len(all_df[all_df['label']=='excluded'])} total 'excluded' verdicts, "
          f"{len(excluded_only)} after excluding threshold/comparison criteria (age, numeric "
          f"cutoffs) -- those are excluded from this screen since non-overlap there is expected, "
          f"correct reasoning, not hallucination.")
    print(f"{len(candidates)} flagged with ZERO content-word overlap between "
          f"cited_criterion and patient_text -> {OUT_PATH}")
    print("\nThese need a human read -- word-overlap is a triage heuristic, not a verdict. "
          "Some may be legitimate paraphrases (false positives, e.g. a criterion citing "
          "'radiotherapy' when the vignette says 'radiation'); some real hallucinations may "
          "not get flagged if they happen to share a word with the text (false negatives).")
    print("\nReading tip (no clinical background needed for this pass): for each flagged row, "
          "just ask 'does the vignette mention this topic AT ALL, in any form?' If genuinely "
          "nothing in the text relates to the cited fact (like the pregnancy case), that's a "
          "real hallucination candidate. If it's mentioned but in different words, or the model "
          "misapplied a real criterion to the wrong part of the vignette, that's a different, "
          "milder error type -- still worth noting, just not fabrication.")

    if len(candidates):
        print(f"\nfirst few flagged:")
        for _, r in candidates.head(5).iterrows():
            print(f"  [{r['combo']}] query={r['query_id']} doc={r['doc_id']}  "
                  f"cited: \"{r['cited_criterion'][:80]}\"")


if __name__ == "__main__":
    main()
