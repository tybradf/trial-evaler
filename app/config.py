"""
Single source of truth for which judge configs the app exposes. Add a new
model here and it automatically appears in the UI toggle, the live-demo
routing, and the dashboard comparison.

Each entry's `tradeoff_*` fields are the actual held-out (2022) numbers
from the eval harness (see docs/findings.md), not marketing copy. They're
rendered directly in the UI toggle so the choice is made with real numbers
in view, not just a model name.
"""

MODEL_CONFIGS = {
    "sonnet_zero_shot": {
        "display_name": "Sonnet (zero-shot)",
        "model_key": "sonnet",
        "strategy": "zero_shot",
        "positioning": "Avoids costly errors",
        "summary": "Prioritizes avoiding the dangerous error and more often defers judgement.",
        "tradeoff_missed_exclusion_rate": 0.015,
        "tradeoff_abstention_rate": 0.393,
        "tradeoff_cost_per_1k_usd": 6.39,
        "tradeoff_note": "On a 200-case held-out sample per model "
                          "misses a true exclusion in ~1.5% of "
                          "excluded cases (vs Haiku's 5.0%, about 3x safer, "
                          "significant in the one-sided direction. "
                          "Routes "
                          "39.3% of cases to a human reviewer instead "
                          "of committing to an answer.",
    },
    "haiku_zero_shot": {
        "display_name": "Haiku (zero-shot)",
        "model_key": "haiku",
        "strategy": "zero_shot",
        "positioning": "Decisive / low-cost",
        "summary": "Commits to an answer more often and costs less per call, "
                    "yet has a significantly higher rate of "
                    "the dangerous error (false negative true exclusions).",
        "tradeoff_missed_exclusion_rate": 0.050,
        "tradeoff_abstention_rate": 0.315,
        "tradeoff_cost_per_1k_usd": 2.53,
        "tradeoff_note": "On a 200-case held-out sample per model "
                          "misses a true exclusion in ~5.0% of "
                          "excluded cases, but routes only 31.5% of eligible "
                          "cases to a human reviewer, vs Sonnet's 39.3%.",
    },
}

DEFAULT_MODEL = "sonnet_zero_shot"

# Shown in the header's info-icon popover on every page. Single source of
# truth -- update here, not in the templates.
PROJECT_BLURB = (
    "An evaluation frameworking for LLM-based clinical trial matching, scored against "
    "real physician eligibility decisions from a NIH benchmark. All "
    "underlying code, data, retrieval, judge evaluation, and statistical "
    "validation is open source."
)
GITHUB_REPO_URL = "https://github.com/tybradf/trial-evaler"
