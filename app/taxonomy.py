"""
Structured version of the manually-identified taxonomy findings from
docs/findings.md Day 4 -- kept separate from the automated pipeline
because these were found through human reading, not a script, and
hand-maintaining a short list is more honest than pretending a heuristic
found all of them.

Keyed by "{query_id}_{doc_id}". Add to this dict as manual review turns up
more confirmed cases.
"""

MANUAL_TAGS = {
    "33_NCT01223885": [
        {"tag": "confirmed_hallucination",
         "note": "Model cited pregnancy as the exclusion reason; the vignette "
                  "never mentions pregnancy status in any form."},
    ],
    "1_NCT00643591": [
        {"tag": "criterion_scope_misapplied",
         "note": "Model's rationale treats the patient's spinal radiation as "
                  "equivalent to the criterion's actual scope (head/neck/brain "
                  "radiation) -- the criterion is real and correctly quoted, "
                  "but misapplied to the wrong body region."},
    ],
    "24_NCT02485808": [
        {"tag": "label_rationale_contradiction",
         "note": "Rationale explicitly concludes gross hematuria is absent and "
                  "no exclusion applies, but the structured label says "
                  "excluded anyway."},
    ],
}


def get_tags(query_id, doc_id) -> list:
    return MANUAL_TAGS.get(f"{query_id}_{doc_id}", [])
