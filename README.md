# Clinical Trial Matching: An Eval Harness for RAG Pipelines

**A retrieve-then-judge LLM pipeline for clinical trial eligibility misses a true exclusion in about 1 of every 120 cases on held-out data — down from 1 in 15 for a 2.5x cheaper model. Raw accuracy alone would have missed this entirely; the two models are statistically tied on it.**

This is a benchmark, not a product pitch: it scores a real retrieve-then-judge pipeline against physician ground truth on real clinical trials, and the deliverable is the eval harness — the retrieval comparison, the 3-class judge evaluation, the error taxonomy, the clinically-weighted cost metric — not the pipeline itself, which is a reference implementation used to exercise the benchmark.

**[Full findings log →](docs/findings.md)** · Live demo: *(add your deployed URL here once live — see `app/README.md` for deployment)*

---

## The headline, in full

Scored against **physician-adjudicated eligibility judgments** from the TREC Clinical Trials track (2021 dev / 2022 held-out test), two configurations of the same judge — Claude Sonnet and Claude Haiku, both zero-shot — land within 3 points of each other on raw accuracy (44.2% vs. 46.7%). If that were the only number examined, they'd look interchangeable, and the cheaper model would win on price with no apparent cost.

It isn't, and it doesn't. On the error that actually matters clinically — telling a patient they're eligible for a trial the physician excluded them from — **Sonnet misses 1 in 120; Haiku misses 1 in 15, an 8x gap that held up on data neither model was tuned on.** The real trade isn't accuracy for cost; it's a materially safer error profile for roughly $4 more per 1,000 patient-trial pairs — a difference that's trivial in absolute terms at any real deployment scale.

The honest cost of the safer model: it hedges more. 44% of eligible-ground-truth cases get routed to `insufficient_information` rather than a direct answer, vs. 30% for the cheaper model. That's a real operational cost — more cases requiring human review — stated plainly rather than glossed over.

## What's actually in here

- **Real data throughout, one deliberate exception.** ~48,700 real ClinicalTrials.gov trials, physician-adjudicated relevance judgments from TREC, and a live ClinicalTrials.gov API v2 connection for the demo. The one synthetic component — patient vignettes — is TREC's own design choice (real patient records can't be public), not this project's.
- **A retrieval comparison with a real, surprising finding.** General-purpose embeddings (`all-mpnet-base-v2`) beat a purpose-built biomedical model (NIH's MedCPT) by roughly 2x on nDCG@10 and recall — and the reason isn't "general models are just better," it's that MedCPT was trained on real PubMed search-log queries and specifically failed on textbook-style synthetic vignettes, a genre it never saw in training. Domain match on *content* isn't the same as domain match on *register*.
- **A judge evaluated on the error that matters, not just accuracy.** A custom clinically-weighted cost metric (a missed exclusion costs 5x an over-exclusion, which costs 2x a hedge) that changes which model looks better — and is the reason it does.
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
