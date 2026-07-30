# Clinical Trial Matching: An Eval Harness for RAG Pipelines

**A retrieve-then-judge LLM pipeline for clinical trial eligibility misses a true exclusion about twice as often with a cheaper model as with a safer one — a real, statistically significant gap (p=0.049 on a 500-case held-out sample), but a modest one, not the dramatic 8x gap an earlier, smaller sample suggested. Raw accuracy alone would point you at the wrong model for the reason that actually matters: it now favors the less-safe model.**

This is a benchmark, not a product pitch: it scores a real retrieve-then-judge pipeline against physician ground truth on real clinical trials, and the deliverable is the eval harness — the retrieval comparison, the 3-class judge evaluation, the error taxonomy, the clinically-weighted cost metric, and a statistical significance test on the headline claim itself — not the pipeline, which is a reference implementation used to exercise the benchmark.

**[Full findings log →](docs/findings.md)** · Report: `https://<you>.github.io/trial-evaler/` (static, deployed on push via GitHub Actions) · Live demo: linked from the report, running on Render (needs a real server — see [Deployment](#deployment))

---

## The headline, in full

Scored against **physician-adjudicated eligibility judgments** from the TREC Clinical Trials track (2021 dev / 2022 held-out test, 500 pairs per class per model on the final run), raw accuracy actually favors the cheaper model — Haiku at 53.7% vs. Sonnet at 49.0%. If that were the only number examined, you'd pick Haiku and believe you gave up nothing.

You would have given something up. On the error that actually matters clinically — telling a patient they're eligible for a trial the physician excluded them from — **Sonnet misses a true exclusion in 2.6% of at-risk cases; Haiku misses 5.2%, roughly twice as often.** Fisher's exact test on this gap: p=0.049, significant at the conventional threshold, but only just — the 95% confidence intervals still overlap (Sonnet [1.5%, 4.4%], Haiku [3.6%, 7.5%]), so treat the direction as established and the exact magnitude as not yet tightly pinned down.

**This number was revised down from an original 8x estimate, and that revision is part of the actual finding, not a correction to hide.** An initial 120-pair sample suggested Sonnet was 8x safer; running the same test at 1,000 pairs showed that estimate was mostly driven by Haiku drawing an unusually bad small sample, not by Sonnet being exceptionally safe. The true effect is real but roughly a quarter the size first claimed — which is exactly the kind of thing a properly-powered follow-up test exists to catch.

The real trade isn't dollars — Sonnet costs about $4 more per 1,000 pairs than Haiku, trivial at any deployment scale. It's **human review capacity**: Sonnet routes 39.7% of eligible cases to `insufficient_information` instead of a direct answer, vs. Haiku's 31.3% — roughly 84 more cases per 1,000 needing a human reviewer. Human reviewer time almost certainly costs more per unit than LLM tokens, so this is likely the more expensive line item between the two models even without a precise reviewer-capacity number — though a real deployment decision would want that number specifically, not an assumption. Recommendation: Sonnet, on the strength of the (real, if modest) safety edge, with this tradeoff stated explicitly rather than glossed over.

## What's actually in here

- **Real data throughout, one deliberate exception.** ~48,700 real ClinicalTrials.gov trials, physician-adjudicated relevance judgments from TREC, and a live ClinicalTrials.gov API v2 connection for the demo. The one synthetic component — patient vignettes — is TREC's own design choice (real patient records can't be public), not this project's.
- **A retrieval comparison with a real, surprising finding.** General-purpose embeddings (`all-mpnet-base-v2`) beat a purpose-built biomedical model (NIH's MedCPT) by roughly 2x on nDCG@10 and recall — and the reason isn't "general models are just better," it's that MedCPT was trained on real PubMed search-log queries and specifically failed on textbook-style synthetic vignettes, a genre it never saw in training. Domain match on *content* isn't the same as domain match on *register*.
- **A judge evaluated on the error that matters, not just accuracy.** A custom clinically-weighted cost metric (a missed exclusion costs 5x an over-exclusion, which costs 2x a hedge) that changes which model looks better — and is the reason it does.
- **A headline claim that was tested, not just asserted.** Fisher's exact test and Wilson confidence intervals on the missed-exclusion gap, run at two sample sizes — the second run cut the original estimate from 8x down to a real, significant, but more modest ~2x, and that revision is documented as a finding, not quietly patched out.
- **An error taxonomy built from real adjudication, not keyword heuristics.** Includes a confirmed hallucination (the judge cited "pregnant" as an exclusion reason for a patient whose vignette never mentions pregnancy), a criterion-scope misapplication (spinal radiation treated as equivalent to head/neck/brain radiation), a label/rationale self-contradiction, and a documented, honest account of a hallucination-detection heuristic's own false-positive rate and how it was fixed.
- **A held-out test that actually held out.** Dev-set tuning happened only on 2021; the reported numbers above are 2022, touched for the first time after the config was locked.

## Try it

A Flask app with three views: a precomputed eval dashboard (loads instantly, no API calls), a case browser (real held-out pairs with every judge verdict alongside physician ground truth, including the hallucination case), and a live demo (real ClinicalTrials.gov search + live judgment, with a model toggle between the safety-first and cost-first configs described above).

```bash
cd app
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python data_prep.py   # bundles the real eval data (ships with a small sample otherwise)
python app.py
```

See `app/README.md` for deployment (Render config included).

## Deployment

Two pieces, deployed separately, because they have genuinely different requirements:

- **The report** (`/` and `/explore`) — pure precomputed data, no secrets, no server needed. Exported to
  static HTML and published to **GitHub Pages** by `.github/workflows/deploy-pages.yml` on every push to
  `main`.
- **The live demo** (`/live`) — needs a running Python process and a secret `ANTHROPIC_API_KEY`, which GitHub
  Pages cannot provide (it only serves static files). Deployed to **Render** by
  `.github/workflows/deploy-render.yml`, which triggers Render's deploy hook on push.

Both are triggered by GitHub Actions on push, matching the "push to `main`, it's live" workflow — they just
publish to two different places because only one of them can be static.

**One-time setup:**
1. Create the Render web service (New → Blueprint → point at this repo; it reads `render.yaml`). Set
   `ANTHROPIC_API_KEY` in Render's dashboard.
2. Copy Render's Deploy Hook URL (service Settings page) into a GitHub repo secret named
   `RENDER_DEPLOY_HOOK_URL`.
3. Add a GitHub repo **variable** (not secret — it's just a URL) named `LIVE_APP_URL` set to your Render
   app's `/live` URL, e.g. `https://trial-evaler.onrender.com/live`. The static site's "Live demo" link uses
   this.
4. Enable GitHub Pages: repo Settings → Pages → Source → "GitHub Actions."
5. Push to `main`. Both workflows run; the report appears at `https://<you>.github.io/<repo>/` and the live
   demo at your Render URL.

## Repo structure

```
src/data_prep/    Day 1 — TREC topics/qrels, ClinicalTrials.gov corpus, eligibility parsing
src/retrieval/    Day 2 — dual embeddings, FAISS, retrieval eval
src/judge/        Day 3 — grounded LLM judge, prompts, sampling
src/eval/         Day 4-5 — 3-class eval harness, error taxonomy, cost/quality frontier
app/              Day 6-7 — Flask UI, deployment config
docs/findings.md  Full chronological findings log — every bug, every dead end, every number
```

## Methodology notes worth knowing before reading the numbers

- **Eligible vs. excluded is not a symmetric task.** Asserting "eligible" requires confirming every inclusion criterion and ruling out every exclusion criterion from a ~150-300 token vignette; asserting "excluded" requires finding one disqualifying fact. This structural asymmetry, not model weakness, is the main driver of the eligible-class recall numbers throughout — confirmed by testing whether it holds for both models (it does) and whether it's specific to any one prompt strategy (it isn't).
- **Cross-model agreement was used to separate real errors from likely ground-truth noise.** When multiple independent models agree with each other but disagree with the physician judgment on the same pair, that's evidence the vignette is missing chart context the physician had access to — not evidence the model is wrong. This distinction is made explicit in the taxonomy rather than folded into a single error rate.
- **Not every automated finding survived a second look.** A first-pass hallucination-detection heuristic flagged 64 cases; manual review showed nearly all were false positives from a fixable bug (age/threshold criteria falsely flagged as ungrounded). The fix, and the honest account of what the tool did and didn't catch, is in `docs/findings.md` — a first-pass tool being wrong at low precision and then getting fixed is more representative of real eval work than a tool that was right the first time.

Full chronological detail — every bug found, every hypothesis tested and sometimes rejected, every number and how it was checked — is in [`docs/findings.md`](docs/findings.md).
