# App

Flask UI for the trial-matching eval project. Three pages:

- **`/` — Eval report.** Precomputed, loads instantly, no API calls. Headline
  numbers, retrieval comparison, judge comparison, confusion matrix.
- **`/explore` — Case browser.** Real held-out (2022) patient/trial pairs
  with full text, criteria, and every judge model's verdict side by side
  against physician ground truth. Includes cases the model got wrong on
  purpose (e.g. the confirmed hallucination case) for full transparency.
- **`/live` — Live demo.** Free-text patient description, live search
  against currently-recruiting ClinicalTrials.gov trials, live judge call.
  Real time inference without ground truth.

## Model selection

`config.py` is the single source of truth for which judge configs the app
exposes (currently `sonnet_zero_shot` and `haiku_zero_shot`, the two
configs actually run through the full eval harness). Each entry's
`tradeoff_*` fields are real held-out numbers from `docs/findings.md`, not
marketing copy -- they render directly in the UI toggle on both `/` and
`/live`. To add a third model: add an entry to `MODEL_CONFIGS`, nothing
else in the app needs to change.

## Setup

```bash
pip install -r ../requirements.txt   # from repo root, or from app/ with the path adjusted
export ANTHROPIC_API_KEY=your_key_here

# Bundle real eval data (run once, and again any time the underlying data changes)
python data_prep.py

# Run
python app.py
```

Visit `http://localhost:5000`.

**Before running `data_prep.py` for the first time**, `app/data/dashboard_data.json`
ships with a small real-numbers sample (3 explorer cases, real dev/test
metrics) so the dashboard and explore pages work out of the box for a
quick look. Run `data_prep.py` to replace it with full pipeline's
actual output (all held-out pairs, full confusion matrices, etc.).

## To Dos:

- **Rate limiting** on `/api/live_judge` and `/api/live_search` -- both
  call paid/external APIs with no cap as written.
- **`live_search.py`'s field/param names** (`query.cond`,
  `filter.overallStatus`) are per the ClinicalTrials.gov API v2 docs as of
  when this was written.

## Deployment

The app splits into two deployable pieces -- see the top-level README's
"Deployment" section for the full one-time setup:

- `/` and `/explore` are pure precomputed data with no secrets, so they're
  exported to static HTML (`python app/build_static.py --live-url ...`)
  and published to GitHub Pages via `.github/workflows/deploy-pages.yml`.
- `/live` needs a running server and a secret `ANTHROPIC_API_KEY`, deploys to Render via
  `.github/workflows/deploy-render.yml` + `render.yaml` instead.

To build and preview the static export locally (without the live demo):

```bash
python app/build_static.py --live-url https://your-app.onrender.com/live
python -m http.server 8000 --directory app/static_build
```
