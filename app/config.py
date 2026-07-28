"""
Single source of truth for which judge configs the app exposes. Add a new
model here and it automatically appears in the UI toggle, the live-demo
routing, and the dashboard comparison -- nothing else needs to change.

Each entry's `tradeoff_*` fields are the actual held-out (2022) numbers
from the eval harness (see docs/findings.md), not marketing copy -- they're
rendered directly in the UI toggle so the choice is made with real numbers
in view, not just a model name.
"""

MODEL_CONFIGS = {
    "sonnet_zero_shot": {
        "display_name": "Sonnet (zero-shot)",
        "model_key": "sonnet",
        "strategy": "zero_shot",
        "positioning": "Safety-first",
        "summary": "Prioritizes avoiding the dangerous error -- willing to say "
                    "\"I don't know\" rather than guess wrong.",
        "tradeoff_missed_exclusion_rate": 0.008,
        "tradeoff_abstention_rate": 0.442,
        "tradeoff_cost_per_1k_usd": 6.52,
        "tradeoff_note": "On held-out data: misses a true exclusion in ~1 of every "
                          "120 cases, but routes 44% of eligible cases to a human "
                          "reviewer instead of committing to an answer.",
    },
    "haiku_zero_shot": {
        "display_name": "Haiku (zero-shot)",
        "model_key": "haiku",
        "strategy": "zero_shot",
        "positioning": "Decisive / low-cost",
        "summary": "Commits to an answer more often and costs less per call -- "
                    "at a materially higher rate of the dangerous error.",
        "tradeoff_missed_exclusion_rate": 0.067,
        "tradeoff_abstention_rate": 0.300,
        "tradeoff_cost_per_1k_usd": 2.59,
        "tradeoff_note": "On held-out data: misses a true exclusion in ~1 of every "
                          "15 cases (8x Sonnet's rate), but routes 30% of eligible "
                          "cases to a human reviewer instead of 44%.",
    },
}

DEFAULT_MODEL = "sonnet_zero_shot"
