"""
Flask app for the trial-matching eval project. Three views:

  /            eval dashboard -- precomputed, loads instantly, no API calls
  /explore     benchmark explorer -- browse real held-out pairs with judge
               verdicts side by side against physician ground truth
  /live        live demo -- free-text patient description, live retrieval
               against current ClinicalTrials.gov recruiting trials, live
               judge call. Rate-limited, clearly labeled as no ground truth.

Model selection (Sonnet vs Haiku, see config.py) is a single dropdown that
drives both the /live judge call and which combo's verdict is highlighted
by default in /explore -- change MODEL_CONFIGS in config.py to add another
model; nothing else in this file needs to change.

Run locally:
    export ANTHROPIC_API_KEY=...
    python app/app.py
"""
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, url_for

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "judge"))
sys.path.insert(0, str(REPO_ROOT / "src" / "retrieval"))

from config import MODEL_CONFIGS, DEFAULT_MODEL, PROJECT_BLURB, GITHUB_REPO_URL

app = Flask(__name__)

DASHBOARD_DATA = json.loads((APP_DIR / "data" / "dashboard_data.json").read_text())

# Live-demo dependencies (embedder, FAISS index, judge client) are loaded
# lazily on first /live request, not at import time -- the dashboard and
# explorer views should work even before torch/anthropic are configured,
# since those two are pure precomputed-data pages.
_live_deps = {"loaded": False}


def _ensure_live_deps():
    if _live_deps["loaded"]:
        return
    from judge_client import judge_pair  # noqa
    _live_deps["judge_pair"] = judge_pair
    _live_deps["loaded"] = True


def _base_context(active_page: str) -> dict:
    """Shared template context for nav/assets -- same shape the static
    exporter (build_static.py) provides, so templates don't need to know
    whether they're being served dynamically or exported to static HTML."""
    return {
        "nav": {
            "dashboard": url_for("dashboard"),
            "explore": url_for("explore"),
            "live": url_for("live"),
        },
        "asset_prefix": url_for("static", filename=""),
        "active_page": active_page,
        "is_static_build": False,
        "project_blurb": PROJECT_BLURB,
        "github_url": GITHUB_REPO_URL,
    }


@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        data=DASHBOARD_DATA,
        models=MODEL_CONFIGS,
        **_base_context("dashboard"),
    )


@app.route("/explore")
def explore():
    return render_template(
        "explore.html",
        pairs=DASHBOARD_DATA["explorer_pairs"],
        models=MODEL_CONFIGS,
        default_model=DEFAULT_MODEL,
        **_base_context("explore"),
    )


@app.route("/live")
def live():
    return render_template(
        "live.html",
        models=MODEL_CONFIGS,
        default_model=DEFAULT_MODEL,
        **_base_context("live"),
    )


@app.route("/api/live_search", methods=["POST"])
def api_live_search():
    """Live ClinicalTrials.gov search -- proves the demo runs against
    current, real recruiting trials, not the frozen benchmark corpus."""
    from live_search import search_recruiting_trials

    payload = request.get_json(force=True)
    condition = (payload.get("condition") or "").strip()
    if not condition:
        return jsonify({"error": "condition is required"}), 400

    try:
        results = search_recruiting_trials(condition, max_results=5)
    except Exception as e:
        return jsonify({"error": f"ClinicalTrials.gov search failed: {e}"}), 502

    return jsonify({"results": results})


@app.route("/api/live_judge", methods=["POST"])
def api_live_judge():
    """Judge a free-text patient description against one trial (selected
    from /api/live_search results). Rate-limiting note: add a per-session
    or per-IP cap before public deployment -- this endpoint calls a paid
    API with no limit as written."""
    payload = request.get_json(force=True)
    patient_text = (payload.get("patient_text") or "").strip()
    trial_title = payload.get("trial_title", "")
    inclusion = payload.get("inclusion_criteria", [])
    exclusion = payload.get("exclusion_criteria", [])
    model_config_key = payload.get("model_config", DEFAULT_MODEL)

    if not patient_text:
        return jsonify({"error": "patient_text is required"}), 400
    if model_config_key not in MODEL_CONFIGS:
        return jsonify({"error": f"unknown model_config '{model_config_key}'"}), 400

    cfg = MODEL_CONFIGS[model_config_key]

    try:
        _ensure_live_deps()
        result = _live_deps["judge_pair"](
            patient_text=patient_text,
            trial_title=trial_title,
            inclusion=inclusion,
            exclusion=exclusion,
            model_key=cfg["model_key"],
            strategy=cfg["strategy"],
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
