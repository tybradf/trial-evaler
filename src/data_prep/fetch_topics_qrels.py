"""
Pull the TREC Clinical Trials 2021 (dev) and 2022 (held-out test) patient
topics and physician relevance judgments (qrels).

Uses the `ir_datasets` package, which wraps NIST's own download + parsing
for this exact benchmark (dataset ids: clinicaltrials/2021/trec-ct-2021 and
clinicaltrials/2021/trec-ct-2022 -- both years share the same underlying
April-2021 ClinicalTrials.gov document collection, only the topics/qrels
differ, per ir-datasets.com/clinicaltrials.html).

Run locally (NOT in a sandboxed environment -- this needs unrestricted
internet access to reach ir_datasets' download servers, which mirror the
NIST files):

    pip install -r requirements.txt
    python src/data_prep/fetch_topics_qrels.py

Relevance scale (confirmed from the raw qrels file):
    0 = not relevant, 1 = excluded, 2 = eligible
"""
import io

import pandas as pd
import requests
import ir_datasets
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET_IDS = {
    2021: "clinicaltrials/2021/trec-ct-2021",
    2022: "clinicaltrials/2021/trec-ct-2022",
}

# Direct NIST source, used as a fallback when an ir_datasets entity doesn't
# expose qrels_iter (as of this writing, the trec-ct-2022 registration only
# wires up queries_iter -- see github.com/allenai/ir_datasets/issues/165).
QRELS_URL_TEMPLATE = "https://trec.nist.gov/data/trials/qrels{year}.txt"


def fetch_qrels_direct(year: int) -> pd.DataFrame:
    """Fallback: download and parse the qrels file straight from NIST.
    Format confirmed against the live file: whitespace-separated
    'query_id iteration doc_id relevance', relevance in {0,1,2}."""
    url = QRELS_URL_TEMPLATE.format(year=year)
    print(f"[{year}] ir_datasets has no qrels for this entity -- "
          f"falling back to direct download: {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(
        io.StringIO(resp.text), sep=r"\s+", header=None,
        names=["query_id", "iteration", "doc_id", "relevance"],
    )


def pull_year(year: int, dataset_id: str) -> None:
    print(f"[{year}] loading {dataset_id} ...")
    ds = ir_datasets.load(dataset_id)

    topics = pd.DataFrame(ds.queries_iter())  # columns: query_id, text
    topics.to_csv(RAW_DIR / f"topics_{year}.csv", index=False)
    print(f"[{year}] {len(topics)} topics -> topics_{year}.csv")

    try:
        qrels = pd.DataFrame(ds.qrels_iter())  # columns: query_id, doc_id, relevance, iteration
    except AttributeError:
        qrels = fetch_qrels_direct(year)

    qrels.to_csv(RAW_DIR / f"qrels_{year}.csv", index=False)
    print(f"[{year}] {len(qrels)} judgments, "
          f"{qrels['doc_id'].nunique()} unique judged trials -> qrels_{year}.csv")


def main():
    for year, dataset_id in DATASET_IDS.items():
        pull_year(year, dataset_id)

    # Union of judged NCT IDs across both years -- this is the trial corpus
    # we actually need to fetch fresh from the ClinicalTrials.gov API v2 in
    # the next step. We do NOT need the full ~375k-trial 2021 XML snapshot
    # that ir_datasets can also provide via ds.docs_iter() -- that's a
    # multi-GB download for trials we'll never use in eval.
    qrels_2021 = pd.read_csv(RAW_DIR / "qrels_2021.csv")
    qrels_2022 = pd.read_csv(RAW_DIR / "qrels_2022.csv")
    unique_ids = sorted(set(qrels_2021["doc_id"]) | set(qrels_2022["doc_id"]))
    (RAW_DIR / "judged_nct_ids.txt").write_text("\n".join(unique_ids))
    print(f"\nTotal unique judged NCT IDs across 2021+2022: {len(unique_ids)}")
    print(f"-> {RAW_DIR / 'judged_nct_ids.txt'}")


if __name__ == "__main__":
    main()
