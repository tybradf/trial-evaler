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
        "tradeoff_missed_exclusion_rate": 0.026,
        "tradeoff_abstention_rate": 0.397,
        "tradeoff_cost_per_1k_usd": 6.34,
        "tradeoff_note": "On a 500-case held-out sample per model: misses a true "
                          "exclusion in ~2.6% of excluded cases (vs Haiku's 5.2% "
                          "-- about 2x safer, statistically significant but not "
                          "dramatically so), while routing 39.7% of eligible "
                          "cases to a human reviewer instead of committing to an "
                          "answer.",
    },
    "haiku_zero_shot": {
        "display_name": "Haiku (zero-shot)",
        "model_key": "haiku",
        "strategy": "zero_shot",
        "positioning": "Decisive / low-cost",
        "summary": "Commits to an answer more often and costs less per call -- "
                    "at a real, if smaller-than-first-estimated, rate of the "
                    "dangerous error.",
        "tradeoff_missed_exclusion_rate": 0.052,
        "tradeoff_abstention_rate": 0.313,
        "tradeoff_cost_per_1k_usd": 2.52,
        "tradeoff_note": "On a 500-case held-out sample per model: misses a true "
                          "exclusion in ~5.2% of excluded cases, but routes only "
                          "31.3% of eligible cases to a human reviewer, vs "
                          "Sonnet's 39.7%.",
    },
}

DEFAULT_MODEL = "sonnet_zero_shot"
