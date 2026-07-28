"""
Rebuild data/processed/corpus.parquet from the already-fetched
data/raw/trials_raw.jsonl -- use this after any change to
parse_eligibility.py, instead of re-hitting the ClinicalTrials.gov API.

    python src/data_prep/reparse_corpus.py
"""
import json
from pathlib import Path

import pandas as pd

from fetch_trials import flatten_study  # reuses the same parsing logic

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def main():
    raw_path = RAW_DIR / "trials_raw.jsonl"
    studies = [json.loads(line) for line in raw_path.open()]
    print(f"Re-parsing {len(studies)} trials from {raw_path} ...")

    rows = [flatten_study(s) for s in studies]
    corpus = pd.DataFrame(rows)
    corpus.to_parquet(PROCESSED_DIR / "corpus.parquet", index=False)

    n_review = corpus["eligibility_needs_review"].sum()
    print(f"Corpus rebuilt -> {PROCESSED_DIR / 'corpus.parquet'}")
    print(f"{n_review} / {len(corpus)} rows flagged eligibility_needs_review "
          f"(was 2740 before the header-regex fix -- should drop now).")


if __name__ == "__main__":
    main()
