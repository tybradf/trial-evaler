"""
Fetch full trial records for every NCT ID that appears in the TREC 2021/2022
qrels, from the live ClinicalTrials.gov API v2 (JSON, no key required).

We deliberately do NOT use the ~375k-trial static 2021 XML snapshot that
ir_datasets can also serve -- we only need the ~26-27k trials that were
actually judged, and pulling them fresh gives us current status/fields for
free (useful context in the writeup: "X% of judged trials have since closed
or completed").

Run locally, after fetch_topics_qrels.py:

    python src/data_prep/fetch_trials.py

NOTE (verify on first run): this assumes the v2 API's `filter.ids` param
accepts a comma-separated batch of NCT IDs per request. That's documented
behavior as of the last time this was checked (see clinicaltrials.gov/data-api
and the migration guide), but the API has changed shape once already (v1->v2
in 2024) -- if batched requests start erroring, drop CHUNK_SIZE to 1 and it
will fall back to one request per trial (slower, ~26k requests, still fine
for a one-time pull with the built-in rate limiting below).
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from parse_eligibility import parse_eligibility_text

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://clinicaltrials.gov/api/v2/studies"
CHUNK_SIZE = 50          # NCT IDs per request; drop to 1 if filter.ids batching fails
REQUEST_DELAY_SEC = 0.3  # be polite to a free public API
FIELDS = [
    "NCTId", "BriefTitle", "OfficialTitle", "Condition", "Phase",
    "OverallStatus", "StudyType", "EligibilityCriteria", "Sex",
    "MinimumAge", "MaximumAge", "HealthyVolunteers", "BriefSummary",
]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def fetch_chunk(nct_ids: list[str]) -> list[dict]:
    params = {
        "filter.ids": ",".join(nct_ids),
        "fields": ",".join(FIELDS),
        "pageSize": len(nct_ids),
        "format": "json",
    }
    resp = requests.get(API_BASE, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("studies", [])


def chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def flatten_study(study: dict) -> dict:
    """Pull the fields we need out of the nested protocolSection structure."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    cond = proto.get("conditionsModule", {})
    elig = proto.get("eligibilityModule", {})
    desc = proto.get("descriptionModule", {})

    elig_text = elig.get("eligibilityCriteria", "")
    parsed = parse_eligibility_text(elig_text)

    return {
        "nct_id": ident.get("nctId"),
        "brief_title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        "conditions": "; ".join(cond.get("conditions", []) or []),
        "phase": "; ".join(design.get("phases", []) or []),
        "overall_status": status.get("overallStatus"),
        "study_type": design.get("studyType"),
        "sex": elig.get("sex"),
        "minimum_age": elig.get("minimumAge"),
        "maximum_age": elig.get("maximumAge"),
        "healthy_volunteers": elig.get("healthyVolunteers"),
        "brief_summary": desc.get("briefSummary"),
        "eligibility_raw": elig_text,
        "inclusion_criteria": parsed["inclusion"],
        "exclusion_criteria": parsed["exclusion"],
        "eligibility_needs_review": parsed["needs_review"],
    }


def main():
    nct_ids = RAW_DIR.joinpath("judged_nct_ids.txt").read_text().splitlines()
    nct_ids = [x.strip() for x in nct_ids if x.strip()]
    print(f"Fetching {len(nct_ids)} judged trials from ClinicalTrials.gov API v2 "
          f"in chunks of {CHUNK_SIZE} ...")

    all_studies, failed_chunks = [], []
    for chunk in tqdm(list(chunked(nct_ids, CHUNK_SIZE))):
        try:
            all_studies.extend(fetch_chunk(chunk))
        except Exception as e:
            failed_chunks.append((chunk, str(e)))
        time.sleep(REQUEST_DELAY_SEC)

    if failed_chunks:
        print(f"\n{len(failed_chunks)} chunks failed after retries -- "
              f"writing to failed_chunks.json for a manual re-run pass.")
        (RAW_DIR / "failed_chunks.json").write_text(
            json.dumps([{"ids": c, "error": e} for c, e in failed_chunks], indent=2)
        )

    print(f"\nFetched {len(all_studies)} / {len(nct_ids)} trials.")

    # Save raw API responses (useful for debugging / re-parsing without
    # re-hitting the API) and the flattened, parsed corpus table.
    with open(RAW_DIR / "trials_raw.jsonl", "w") as f:
        for s in all_studies:
            f.write(json.dumps(s) + "\n")

    rows = [flatten_study(s) for s in all_studies]
    corpus = pd.DataFrame(rows)
    corpus.to_parquet(PROCESSED_DIR / "corpus.parquet", index=False)

    n_review = corpus["eligibility_needs_review"].sum()
    print(f"Corpus saved -> {PROCESSED_DIR / 'corpus.parquet'}")
    print(f"{n_review} / {len(corpus)} rows flagged eligibility_needs_review "
          f"(spot-check these before the judge stage).")


if __name__ == "__main__":
    main()
