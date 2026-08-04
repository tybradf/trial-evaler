# Clinical Trial Matching: An Evaluation Harness for RAG Pipelines

## Overview

Matching patients to clinical trials requires a deep understanding of 1) individual patient's medical conditions and 2) individual trials inclusion and, perhaps more importantly, exclusion criteria. [Trial Matcher](https://trial-evaler.onrender.com/live) is a tool for matching patient chart notes to real clinical trials. The tool offers two LLM options for matching, one safer against the riskiest error (categorizing a patient as eligible when they are not) and the other with a lower abstention rate (i.e., workflow requires less manual review).

This repository builds the retrieve-then-judge LLM pipeline for clinical trial eligibility, including development on data from 2021 and evaluation on holdout data from 2022. The site includes all the findings from model selection and both model's performances on the 2022 hold out data, with the ability to inspect individual cases within each category of the confusion matrix.

**[Full findings log →](docs/findings.md)** 


## Findings summary

Scored against **physician-adjudicated eligibility judgments** from the TREC Clinical Trials track (2021 dev / 2022 held-out test, 500 pairs per class per model on the final run), raw accuracy actually favors the cheaper model — Haiku at 53.7% vs. Sonnet at 49.0%.

But accuracy is not the important evaluation metric. On the error that actually matters most clinically — the LLM telling a patient they're eligible for a trial the physician excluded them from — **Sonnet misses a true exclusion in 2.6% of at-risk cases; Haiku misses 5.2%, roughly twice as often.** A Fisher's exact test on this gap is significant (p=0.049), however the 95% confidence intervals still overlap (Sonnet [1.5%, 4.4%], Haiku [3.6%, 7.5%]). As such , we treat the direction as established, but the exact magnitude could be refined with further testing and larger sample sizes. 

Sonnet is a more expensive model than Haiku. But the real trade isn't model API costs: Sonnet costs about $4 more per 1,000 pairs than Haiku, trivial at any deployment scale. It's **human review capacity**: Sonnet routes 39.7% of eligible cases to `insufficient_information` instead of a direct answer, vs. Haiku's 31.3% — roughly 84 more cases per 1,000 needing a human reviewer. Human reviewer time (likely a physician or other trained clinician) almost certainly costs more per unit than LLM tokens, so this is likely the more expensive line item between the two models.

## Methodology overview

1. **Real data throughout, except patient data** ~48,700 real ClinicalTrials.gov trials, physician-adjudicated relevance judgments from TREC, and a live ClinicalTrials.gov API v2 connection for the live demo. The only synthetic data component is the patient records, sourced from TREC, because real patient notes cannot be public.
1. **Retrieval comparison** General-purpose embeddings (`all-mpnet-base-v2`) beat a purpose-built biomedical model (NIH's MedCPT) by roughly 2x on nDCG@10 and recall. MedCPT was trained on real PubMed search-log queries, not textbook-style synthetic vignettes: a genre it never saw in training. Just because MedCPT is biomedical doesn't mean it fit our use case best.
1. **Development data vs. hold out data.** Dev-set tuning only used 2021 data. Judge performance numbers are based entirely on 2022 data. 
1. **Clinically-weighted cost metrics over pure accuracy.** Instead of accuracy for model selection, use a custom clinically-weighted cost metric (a missed exclusion costs 5x an over-exclusion, which costs 2x a hedge). These weights could certainly be modified with clinician input, and are very influential in model selection. 
1. **Statistical testing on model difference.** Fisher's exact test and Wilson confidence intervals on the missed-exclusion gap, run at two sample sizes. Single-sided Fisher's demonstrated significantly lower misses of true exclusions for Sonnet. 
1. **Error detection based on human adjudication.** Several scripts in `src/eval/` are test built in response to manually adjudicating results from the LLMs, such as a confirmed hallucination (the judge cited "pregnant" as an exclusion reason for a patient whose vignette never mentions pregnancy). These serve only as traige tools.

## Initial set up

A Flask app with three views: 

1. Precomputed evaluation dashboard (loads instantly, no API calls), 
1. Case browser (real held-out pairs with every judge verdict alongside physician ground truth, including the hallucination case), and 
1. Live demo (real ClinicalTrials.gov search + live judgment, with a model toggle between the safety-first and cost-first configs described above).

```bash
cd app
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python data_prep.py   # bundles the real eval data (ships with a small sample otherwise)
python app.py
```

See `app/README.md` for deployment (Render config included).

## Deployment

Two pieces, deployed separately:

- **Precomputed data** (`/app/static/`) — static, precomputed data (from the validation process) are hosted in HTML and published to **GitHub Pages** by `.github/workflows/deploy-pages.yml` on every push to
  `main`.
- **Live demo** (`/live`) — Deployed to **Render** by
  `.github/workflows/deploy-render.yml`, which triggers Render's deploy hook on push. Render hosts Anthropic API keys for live matching of patient charts to clinical trials.

Both deploys are executed by Github Actions when any update is pushed to `main`. 

## Repo structure

```
src/data_prep/    TREC topics/qrels, ClinicalTrials.gov corpus, eligibility parsing
src/retrieval/    Dual embeddings, FAISS, retrieval evaluation
src/judge/        LLM judge, prompts, sampling
src/eval/         3-class evaluation harness (eligible, excluded, insufficient information), error taxonomy, cost/quality frontier
app/              Flask UI, deployment config
docs/findings.md  Full chronological findings log — bugs, hypotheses, etc.
```

## Important methodological notes

- **Determining eligible or excluded is not a symmetric task.** Asserting "eligible" requires confirming every inclusion criterion and ruling out every exclusion criterion from a ~150-300 token vignette; asserting "excluded" requires finding one disqualifying fact. This structural asymmetry, not model weakness, is the main driver of the eligible-class recall numbers throughout. The asymmetry holds for both models and is not specific to any one prompt strategy.
- **Model consensus helps separate real errors from likely ground-truth noise.** When multiple independent models agree with each other but disagree with the physician judgment on the same pair, it suggests that the physcian perhaps had additional context not available to the models (i.e., clinical information beyond the chart note).

For a full choronological development timeline, inluding all hypotheses tested and bugs resolved, see: [`docs/findings.md`](docs/findings.md).
