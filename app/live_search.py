"""
Live search against the ClinicalTrials.gov API v2 for currently-recruiting
trials matching a condition. Used by the /live demo -- this is what proves
the demo is running against real, current data rather than the frozen
benchmark corpus.

Deliberately not FAISS/embedding-based: the live demo searches by condition
keyword directly through the API's own query syntax, which is more than
adequate for "show me a few real recruiting trials for X" and avoids
loading the full embedding stack (torch, sentence-transformers) into what
should be a lightweight Flask app.

NOTE (verify on first deploy): field names and query param names
(`query.cond`, `filter.overallStatus`) are per the v2 API docs as of when
this was written -- same caveat as fetch_trials.py in the data pipeline.
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data_prep"))
from parse_eligibility import parse_eligibility_text  # reuse, don't duplicate

API_BASE = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = [
    "NCTId", "BriefTitle", "Condition", "OverallStatus",
    "EligibilityCriteria", "Sex", "MinimumAge", "MaximumAge",
]


def search_recruiting_trials(condition: str, max_results: int = 5) -> list:
    params = {
        "query.cond": condition,
        "filter.overallStatus": "RECRUITING",
        "fields": ",".join(FIELDS),
        "pageSize": max_results,
        "format": "json",
    }
    resp = requests.get(API_BASE, params=params, timeout=15)
    resp.raise_for_status()
    studies = resp.json().get("studies", [])

    results = []
    for s in studies:
        proto = s.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        elig = proto.get("eligibilityModule", {})
        cond = proto.get("conditionsModule", {})

        elig_text = elig.get("eligibilityCriteria", "")
        parsed = parse_eligibility_text(elig_text)

        results.append({
            "nct_id": ident.get("nctId"),
            "title": ident.get("briefTitle"),
            "conditions": cond.get("conditions", []),
            "inclusion_criteria": parsed["inclusion"],
            "exclusion_criteria": parsed["exclusion"],
        })
    return results
