# Clinical Trial Matching: An Eval Harness for RAG Pipelines

**Status: Day 1 of 7 — data pipeline scaffolded, not yet run.**

A benchmark for evaluating retrieve-then-judge LLM pipelines against
physician ground truth on real clinical trial eligibility matching. The
pipeline (retrieval + LLM judge) is a reference implementation used to
exercise the benchmark — the eval harness, not the pipeline, is the
deliverable. See `docs/plan.md` (TODO) for full scope.

## Data sources (all real, public, no DUA required)

- **TREC Clinical Trials 2021 + 2022** — physician-adjudicated relevance
  judgments (eligible / excluded / not relevant) pairing real ClinicalTrials.gov
  trials against clinically-authored patient case topics. 2021 = dev set,
  2022 = held-out test set. Pulled via `ir_datasets`.
- **ClinicalTrials.gov API v2** — live trial records (title, conditions,
  eligibility criteria, etc.) for every judged NCT ID, fetched fresh.
- **Chia** (stretch goal) — entity-level eligibility-criteria annotations,
  via HuggingFace `bigbio/chia`, for a deeper extraction-eval module if time
  allows.

## Day 1: setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python src/data_prep/fetch_topics_qrels.py   # pulls topics + qrels for both years
python src/data_prep/fetch_trials.py         # pulls + parses the judged trials
```

**Run these locally, not in a restricted/sandboxed container** — both
scripts need unrestricted outbound internet access (ir_datasets' download
servers and the live ClinicalTrials.gov API aren't reachable from typical
sandboxed dev environments' network allowlists).

Expected output after both scripts run:
- `data/raw/topics_2021.csv`, `qrels_2021.csv`, `topics_2022.csv`, `qrels_2022.csv`
- `data/raw/judged_nct_ids.txt` — union of unique judged trial IDs
- `data/raw/trials_raw.jsonl` — raw API responses (for debugging/re-parsing)
- `data/processed/corpus.parquet` — clean corpus table, one row per trial

**First-run checks worth doing** before moving to retrieval:
1. Confirm `filter.ids` batching in `fetch_trials.py` actually returns all
   requested trials per chunk (spot-check a few chunk sizes against the
   response count). Drop `CHUNK_SIZE` to 1 if it doesn't.
2. Check the `eligibility_needs_review` flag rate in the corpus — if it's
   high, the eligibility-text parser needs a look at a few of those raw
   strings (older, pre-2008 trial records are the likeliest offenders).
3. Sanity-check qrels counts against the published TREC numbers: 2021 should
   be ~35,832 judgments / ~26,162 unique trials; 2022 ~35,394 / ~26,585.

## Roadmap

- [x] Day 1 — data pipeline (topics, qrels, corpus)
- [ ] Day 2 — retrieval (dual embeddings, FAISS, Recall@k / nDCG@10)
- [ ] Day 3 — grounded LLM judge (multi-model, multi-prompt)
- [ ] Day 4 — 3-class eval harness, confusion matrix, error taxonomy
- [ ] Day 5 — cost/quality frontier, judge-of-judge validation
- [ ] Day 6 — Flask UI (eval dashboard, benchmark explorer, live demo)
- [ ] Day 7 — deploy, README with headline finding
