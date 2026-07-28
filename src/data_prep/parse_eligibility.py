"""
ClinicalTrials.gov API v2 returns eligibility as a single free-text field
(protocolSection.eligibilityModule.eligibilityCriteria), typically shaped like:

    Inclusion Criteria:

    * Adults aged 18 to 75
    * Confirmed diagnosis of X

    Exclusion Criteria:

    * Pregnant or breastfeeding
    * Prior treatment with Y

This is a known API quirk (see clinicaltrials.gov/data-api) -- there is no
structured inclusion/exclusion split in the API response itself, so we have
to parse it ourselves. This parser is deliberately conservative: if it can't
confidently find an "Exclusion Criteria" boundary, it returns everything as
inclusion and flags the row for manual review rather than silently
mis-splitting it. NOTE: verify this against a handful of real live responses
on first run -- formatting is not perfectly consistent across older trials
(pre-2008 records in particular).
"""
import re

EXCLUSION_HEADER_RE = re.compile(r"exclusion\s*criteria\s*:?|exclusion\s*:", re.IGNORECASE)
INCLUSION_HEADER_RE = re.compile(r"inclusion\s*criteria\s*:?|inclusion\s*:", re.IGNORECASE)
BULLET_RE = re.compile(r"^\s*[\*\-\u2022]\s*|^\s*\d+[\.\)]\s*")


def _split_bullets(block: str) -> list[str]:
    lines = [ln.strip() for ln in block.splitlines()]
    items, current = [], []
    for ln in lines:
        if not ln:
            continue
        if BULLET_RE.match(ln):
            if current:
                items.append(" ".join(current).strip())
            current = [BULLET_RE.sub("", ln).strip()]
        else:
            current.append(ln)
    if current:
        items.append(" ".join(current).strip())
    return [i for i in items if i]


def parse_eligibility_text(raw: str | None) -> dict:
    """Returns {'inclusion': [...], 'exclusion': [...], 'needs_review': bool}."""
    if not raw or not raw.strip():
        return {"inclusion": [], "exclusion": [], "needs_review": True}

    excl_match = EXCLUSION_HEADER_RE.search(raw)
    if not excl_match:
        # No exclusion section found at all -- flag it rather than guess.
        incl_text = INCLUSION_HEADER_RE.sub("", raw, count=1)
        return {
            "inclusion": _split_bullets(incl_text),
            "exclusion": [],
            "needs_review": True,
        }

    incl_block = raw[: excl_match.start()]
    excl_block = raw[excl_match.end():]
    incl_block = INCLUSION_HEADER_RE.sub("", incl_block, count=1)

    inclusion = _split_bullets(incl_block)
    exclusion = _split_bullets(excl_block)
    return {
        "inclusion": inclusion,
        "exclusion": exclusion,
        # Flag rows where parsing produced suspiciously little -- worth a
        # manual spot-check pass before they feed the judge stage.
        "needs_review": len(inclusion) == 0 or len(exclusion) == 0,
    }
