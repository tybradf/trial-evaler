# Findings Log

Running record of decisions, bugs, and results, kept as they happen so the
final report doesn't rely on memory. Each entry: what we found, how we knew,
what we did about it.

---

## 1 — Data pipeline

**Sources.** TREC Clinical Trials 2021 (dev) + 2022 (held-out test), pulled
via `ir_datasets`, with a direct-NIST-download fallback for 2022 qrels since
that entity's `ir_datasets` registration only wires up `queries_iter`, not
`qrels_iter`. Confirmed correct against published TREC numbers: 2021 = 75
topics / 35,832 judgments / 26,162 unique judged trials.

**Corpus.** 48,713 / 48,714 judged trials (union of both years' unique NCT
IDs) fetched fresh from the ClinicalTrials.gov API v2, rather than using the
full ~375k-trial static 2021 snapshot ir_datasets also offers — we only
need the trials that were actually judged.

**Eligibility parser bug.** Initial header-based inclusion/exclusion
splitter required the literal word "Criteria" (e.g. "Exclusion Criteria:").
Some real trials use bare "Inclusion:" / "Exclusion:" headers with no
"Criteria" — these were silently mis-parsed as one undifferentiated block.
Found via manual sampling of flagged rows, fixed by broadening the header
regex. Net effect: flagged-row count dropped from 2,740 → 2,639 (48,713
total, 5.4%).

**What the remaining 5.4% actually is (validated, not assumed).** Sampled
and categorized:
- ~1,030 trials use the older NCI PDQ narrative format (`DISEASE
  CHARACTERISTICS: / PATIENT CHARACTERISTICS: / PRIOR CONCURRENT THERAPY:`),
  common in older oncology trials predating structured
  inclusion/exclusion authoring — no clean split exists in the source data.
- 8 trials have empty eligibility text entirely.
- The remaining ~1,600 are legitimately single-arm/observational trials
  with no exclusion section, or trials with an "Exclusion Criteria:" header
  present but nothing documented after it (genuinely missing in the source,
  not a parser failure).
- Conclusion: the flag is doing its job. These are real data-coverage gaps,
  not bugs, and the flag is carried into the corpus for later stratified
  analysis (e.g. "judge accuracy on modern structured-criteria trials vs.
  older PDQ-narrative trials").

**Eval discipline.** 2021 and 2022 topic IDs both start at 1 — kept the two
years in entirely separate files/pipelines throughout (never merged
without namespacing) to avoid a silent ID-collision bug.

---

## 2 — Retrieval

**Embedding models.**
- `general`: sentence-transformers/all-mpnet-base-v2 — strong
  general-purpose, symmetric SBERT.
- `biomed`: ncbi/MedCPT-Query-Encoder + Article-Encoder — NIH/NCBI's
  purpose-built biomedical retrieval model, asymmetric dual-encoder trained
  on 255M real PubMed search-log query→article pairs. Article encoder given
  (title, body) pairs, matching its title+abstract training format.

**Bug: chronological-sort confound in the first smoke test.** Initial
100-trial sanity check used `corpus.head(100)`. Since `judged_nct_ids.txt`
is alphabetically sorted and NCT IDs are sequential, this silently selected
only the *oldest* ~100 trials in the corpus (1990s NIH cohort studies) —
not a representative sample. A biomed result that looked concerning
(irrelevant match for a headache patient) turned out to be an artifact of
this non-random slice combined with a genuinely low-coverage micro-corpus.
Fixed by switching all smoke tests to `.sample()`.

**Bug: NaN-truthiness in the corpus body-text builder.** `row.get(x) or ""`
does not catch `NaN` (it's truthy in Python), so a handful of trials with
genuinely null `conditions`/`brief_summary`/`eligibility_raw` crashed
`embed_corpus.py` on the full 48.7k-trial run (didn't surface on the small
test slice, which happened to have no nulls). Fixed with an explicit
`isinstance(x, str)` check.

**Bug: MedCPT query truncation (the big one).** `MedCPTEmbedder.encode_queries`
used `max_length=64`, MedCPT's own default for short PubMed search-log
queries. TREC patient topics are full clinical vignettes — measured mean
185 tokens (2021) / 129 tokens (2022), max 309, **100% of topics in both
years exceeded 64 tokens**. Truncation (from the end) was silently
discarding most of each topic's clinically salient content before the
model ever saw it. Fixed by raising to 512 (matching the article encoder,
within the underlying BERT architecture's native limit). Re-embedding only
the topic queries (not the corpus) took seconds.

**Effect of the fix — asymmetric and only partial.**

| year | model   | nDCG@10 (pre-fix) | nDCG@10 (post-fix) |
|------|---------|--------------------|----------------------|
| 2021 | biomed  | 0.212              | 0.219 (barely moved) |
| 2022 | biomed  | 0.098              | 0.270 (nearly 3x)    |

Investigated the asymmetry: neither topic length nor qrels-judgment density
correlated meaningfully with per-topic nDCG for either model in either year
— both hypotheses ruled out.

**Final retrieval metrics** (general vs. biomed, both years, post-fix):

| year | model   | nDCG@10 | recall_eligible@10 | @50   | @100  |
|------|---------|---------|----------------------|-------|-------|
| 2021 | general | 0.461   | 0.078                | 0.252 | 0.370 |
| 2021 | biomed  | 0.219   | 0.024                | 0.082 | 0.134 |
| 2022 | general | 0.492   | 0.090                | 0.247 | 0.355 |
| 2022 | biomed  | 0.270   | 0.038                | 0.106 | 0.154 |

General-purpose embeddings beat the biomedical model roughly 2x on nDCG and
recall, consistently across both years, even after fixing the truncation
bug. General's absolute numbers (~0.46–0.49 nDCG@10) land within the range
of published TREC CT track systems — a plausibility check that passed.

**The real explanation (found via zero-nDCG topic analysis).** 10/75 (2021)
and 13/50 (2022) topics scored a *hard* 0.000 nDCG@10 for biomed while
general scored non-zero on nearly all of the same topics — a qualitatively
different failure (nothing relevant in top-10 at all) from generally-worse
ranking. Reading the zero-topics' full text revealed a genre split:
- Real messy de-identified clinical notes (redaction brackets, colloquial
  style, lab-value dumps) — biomed handles these reasonably.
- Clean, templated, textbook/USMLE-style diagnostic vignettes (e.g. near-
  duplicate alcoholic-pancreatitis cases as separate topics, each building
  to a single classic diagnosis) — biomed fails on these specifically and
  almost completely; general does not.

**Interpretation.** MedCPT was trained exclusively on real PubMed
search-log click data — a researcher's literature query paired with the
article they clicked. It has essentially never seen exam-style clinical
case-presentation prose as an input genre, even though the *content* is
nominally biomedical. This reframes the finding from a flat "domain-specific
loses to general-purpose" into something sharper and more defensible: a
retrieval model's fit depends on matching the input's *genre and register*,
not just its subject domain.

**One exception, worth keeping distinct in the writeup.** 2022 topic 16
(sarcoidosis) scored 0.000 for *both* models — likely a corpus coverage gap
(no good sarcoidosis trial exists in the judged set) rather than a retriever
weakness. Flagged separately from the genre-mismatch failures.

**Decision.** General-purpose embeddings (all-mpnet-base-v2) selected as
the retrieval backbone feeding the judge stage. The biomed comparison
is retained in the eval report as a documented, explained finding, not
pursued further as a candidate for production use.

---

## 3 — Judge

**Setup.** Structured output via forced tool-use (label / cited_criterion /
rationale), not free-text parsing. Sampled 120 dev pairs (60 excluded / 60
eligible) from 2021 qrels only, relevance ∈ {1,2} — relevance=0 pairs
excluded as trivial (wrong disease area, not a real eligibility question).
Running 2 models (Haiku 4.5, Sonnet 5) × 2 prompt strategies (zero-shot,
few-shot) = 480 calls.

**Spot check on Haiku/few-shot (n=15, informal read before the full eval
harness).** 12/15 label matches. Reasoning quality is good: citations are
grounded in real criteria text, negation and numeric thresholds (BMI vs.
cutoff, age vs. range) handled correctly.

Three things worth carrying into the error taxonomy with real
examples attached, not just re-derived from aggregate numbers later:

1. **Two direction-errors, different failure types.** Query 62/NCT01591005:
   model said eligible, truth=excluded — a missed exclusion, the more
   dangerous direction. Query 33/NCT00910637: model said excluded,
   truth=eligible — an over-literal reading of "regular cyclic menstrual
   periods" vs. the patient's irregular cycle; the model's logic is
   defensible on the text alone, but ground truth disagrees, suggesting
   either physician context beyond the topic text or genuine model
   over-rigidity.

2. **Repeated hedge-on-eligible pattern (n=2/15, not a one-off).**
   Queries 48 and 7 both returned `insufficient_information` citing a
   specific missing lab value (dermatophyte species, MELD score) where
   ground truth is eligible anyway. Same structure twice in 15 samples.
   Open question for the full eval: is this judge over-caution, or do
   TREC physician annotators treat unstated-but-presumably-normal values
   more permissively than a strict criteria reading would? Worth explicit
   tracking as an abstention-rate metric, not just accuracy.

3. **Label/rationale self-consistency bug, found and now checked
   automatically.** Query 41's rationale text explicitly says "making
   eligibility indeterminate," but the structured label was "excluded,"
   not "insufficient_information" — the model's own stated reasoning
   contradicted its chosen label. `check_label_consistency.py` scans the
   full result set for this (rationale contains hedge language, label is
   not insufficient_information) so it's tracked as its own failure
   category, distinct from factual disagreement with ground truth.

**Cross-model comparison on identical pairs (n=10, Haiku/few-shot vs.
Sonnet/zero-shot on the exact same 10 patient-trial pairs).**

Two disagreements with ground truth appeared in *both* models independently:
- Query 48/NCT00835510 (missing dermatophyte species): both models hedge
  with `insufficient_information`; ground truth = eligible.
- Query 33/NCT00910637 (irregular menses vs. "regular cyclic periods"
  criterion): both models say excluded; ground truth = eligible.

Two structurally different models agreeing with each other but disagreeing
with the physician judgment, on the same two pairs, is real evidence these
are candidate ground-truth-ambiguity cases (physician likely had chart
context beyond the topic text) rather than model errors -- flagged as a
distinct category in the taxonomy, not counted against the models.

**Most interesting single data point: query 62/NCT01591005.** Haiku
confidently answered `eligible` (wrong, dangerous direction -- ground truth
is excluded, a missed exclusion). Sonnet, on the identical pair, returned
`insufficient_information` instead of committing to either label -- a
capability-linked safety behavior: the stronger model recognized its own
uncertainty and declined to guess, where the cheaper model guessed
confidently and wrong, in the costly direction.

**Pattern across all 10:** Sonnet hedged more often overall (3/10 vs.
Haiku's 1/10) but made zero dangerous-direction errors on this set vs.
Haiku's one. A real precision-vs-caution tradeoff -- Sonnet is less often
flatly "correct" but arguably safer -- exactly what the asymmetric-cost
framing in the full eval harness is designed to quantify.

## 4 — Full eval harness

**Headline numbers (n=120 per combo, 2021 dev set):**

| combo | accuracy_strict | mean_cost | missed_exclusion_rate | abstention_rate |
|---|---|---|---|---|
| haiku_zero_shot | 0.558 | 0.550 | 0.050 | 0.183 |
| haiku_few_shot | 0.592 | 0.496 | 0.042 | 0.158 |
| sonnet_zero_shot | 0.592 | 0.283 | 0.000 | 0.250 |
| sonnet_few_shot | 0.558 | 0.304 | 0.000 | 0.275 |

**Accuracy alone would say Haiku and Sonnet are tied.** They aren't. Sonnet's
mean_cost is roughly half Haiku's, and Sonnet made zero dangerous
missed-exclusion errors across 240 judgments vs. Haiku's 11 (~4-5% of
cases). Sonnet trades some additional hedging for reliably avoiding the one
error type that actually matters clinically. Headline point for the report:
*which metric you lead with changes the conclusion* -- raw accuracy and the
clinically-weighted cost metric disagree about which model "wins," and the
cost-weighted one is the more defensible choice given the project's stated
premise that a missed exclusion and an over-exclusion are not equally bad.

**Ground-truth-ambiguity flag (>=75% of combos disagree with qrels): 46/120
pairs (38%).** Too high to take at face value as "ground truth is wrong" --
split into three sub-patterns to find out what's actually driving it:

- `confident_disagreement` (14): all/most combos confidently agree on the
  same wrong label. The strongest, most specific signal -- independent
  models converging on the same wrong answer, worth reading individually.
- `mixed` (20): some combos hedge, some confidently wrong.
- `consensus_hedge` (12): all combos hedge with insufficient_information.
  Consistent with the vignette genuinely lacking a detail the physician's
  full chart had -- smaller share than expected, not the dominant pattern.

**Underlying structural driver of the eligible-class miss rate:**
`recall_eligible` is 0.28-0.33 across all four combos vs. `recall_excluded`
at 0.80-0.87. Asserting "eligible" requires positively confirming every
inclusion criterion and ruling out every exclusion criterion; asserting
"excluded" requires finding just one disqualifying fact. A short vignette
is far more likely to contain that one fact than to exhaustively cover
everything needed to confirm eligibility -- a structural property of the
task/benchmark, not a bug.

**Manual tagging scope, right-sized:** not 100 rows. 8 genuine single-combo
errors (isolated, real model mistakes) + the 14 confident_disagreement
pairs (most informative of the ambiguous bucket) = 22 rows worth reading
closely. The 20 mixed / 12 consensus_hedge pairs are lower priority.

*(confident_disagreement read-through: TODO)*

**Confident_disagreement read-through (n=14, all ground_truth=eligible,
all models confidently said excluded).** Not one explanation -- three
distinct patterns, worth separate error-taxonomy categories:

1. **Likely TREC ground-truth noise, model reasoning is clinically sound**
   (query 12/NCT00807040, 14/NCT03835741, 68/NCT02159079, arguably
   3/NCT00760903). In each, the model's exclusion cites a real, specific,
   correct clinical distinction: Marfan's/structural MR vs. the trial's
   required ischemic MR; BiPAP is literally NIV, an explicit exclusion;
   hemodialysis is literally "renal replacement therapy," an explicit
   exclusion. Hard to construct a chart-context story that changes these.
   Expected base rate of noise in any large pooled human-judgment set --
   worth stating plainly rather than assuming every disagreement is a
   model failure.

2. **Over-literal reading missing clinical intent** (query 63/NCT02351492:
   "radiologically proven CBD stone" as exclusion vs. "choledocholithiasis
   in ultrasound" as inclusion -- likely distinguishing confirmatory-grade
   imaging from the ultrasound that qualifies someone for the trial's own
   better-imaging intervention, a subtlety the model missed; query
   26/NCT01247857: "free from pain in preoperative period" read as "ever
   had pain in this note" rather than "pain-free by the time of the
   scheduled elective surgery," missing that her cholecystectomy was
   explicitly planned as elective/interval, not acute). Real, explainable
   model limitations -- not ground-truth noise.

3. **Vignette plausibly lacks chart context the physician had** (query
   36/NCT00105066: metabolic syndrome criteria partially unmet/unstated
   from the numbers given, e.g. waist circumference never measured; query
   60/NCT03372096: trial requires prior BPH med trial + IPSS + Qmax, none
   mentioned in vignette; query 16/NCT01399983: patient refused RHC *this
   admission* but may have a prior RHC-confirmed diagnosis not mentioned).

**Exclusion-list-length hypothesis (proposed as "maybe longer lists give
the judge more surface area to falsely trigger on"):** not well supported
on this n=14 subset. NCT01223885 (1 exclusion criterion) is one of the
strangest disagreements (vignette doesn't mention the trial's condition at
all), while several of the longest lists (21, 15 items) are exactly the
cases where the model's citation was a specific, correct, real clinical
match -- more signal, not more noise. Full quantitative check across all
120 pairs (`check_exclusion_length_hypothesis.py`) pending to confirm at
scale rather than from 14 eyeballed cases.

**Label/rationale consistency check, full run (480 calls, all 4
model×strategy combos).** 3/120 flagged, and all 3 concentrated in
`haiku_few_shot` — zero in `haiku_zero_shot`, zero in either sonnet combo.
Not random distribution. Likely cause: the few-shot prompt's second example
(MELD-score case) explicitly models hedging language as part of
demonstrating the `insufficient_information` label; plausible that the
cheaper model picks up the hedging *phrasing* without reliably pairing it
with the correct *label*, while Sonnet doesn't show this under either
prompt. Low rate (2.5%), isolated to one combo -- kept as a documented
finding rather than treated as a bug to fix, and worth folding into the
cost/quality frontier as a self-consistency dimension, not just
accuracy-per-dollar.

**Correction to the exclusion-list-length section above: query
33/NCT01223885 is not ground-truth noise -- it's a confirmed
hallucination.** The trial's only exclusion criterion is "Pregnant women";
the vignette never mentions pregnancy status at all (only "uses barrier
methods of contraception," which if anything mildly suggests the
opposite). The model cited pregnancy as the exclusion reason anyway -- not
a hedge, not an inference, a specific asserted fact absent from the source
text. Qualitatively worse than the other error types found so far:
confident and specific, rather than uncertain.

Built `check_hallucinated_citations.py` to screen all "excluded" verdicts
for this pattern systematically: flags any cited_criterion sharing zero
content words with the patient vignette it was judged against. Verified
against this exact case (correctly flags it, zero overlap) and a
legitimate control (BMI citation, correctly not flagged, real overlap).
Heuristic triage tool, not a verdict -- flagged rows still need a human
read.

*(full hallucination-screen results across all 480 judgments: TODO)*

**First version of the hallucination screen had a systematic false-positive
bug, caught by reading the flagged output rather than trusting the count.**
64/290 flagged, but every one of the first several rows inspected was a
correctly-reasoned age/threshold citation (e.g. "Age >70 years" cited
against a 48-year-old patient) -- not a hallucination at all. Root cause:
the word-overlap check stripped digits before comparing, and for
threshold-style criteria (age cutoffs, lab-value comparisons) non-overlap
between the cited threshold and the patient's actual value is *supposed*
to happen -- that mismatch is exactly what makes it a valid exclusion.
Fixed by excluding threshold/comparison-style criteria (detected via
comparison operators, numeric ranges, or "age" + a digit) from the
screen entirely, leaving only categorical-fact citations (pregnant,
smoker, specific diagnosis, etc.) where zero overlap is actually
suspicious. Re-run pending on the full dataset with the fix.

Worth keeping in the report as-is, not smoothing over: a first-pass eval
tool producing a plausible-looking but wrong number, caught by reading a
sample of its own output before trusting it, is a good demonstration of
the "evaluate your evaluator" discipline the project is built around.

**Related, distinct finding surfaced along the way: anatomical/criterion
scope misapplication (query 1/NCT00643591).** The model cited a real
exclusion criterion ("no previous radiotherapy to the head and neck and
brain area") but its own rationale text quietly treated the patient's
spinal radiation as equivalent ("prior radiotherapy to the spine/brain
area..."), when spine is not head/neck/brain. Not a fabrication -- the
criterion is real and correctly quoted -- but a misapplication of a real
criterion to the wrong body region. A third, milder error category
distinct from both hallucination and ground-truth noise; tagged separately
in the taxonomy as `criterion_scope_misapplied`.

**Hallucination screen closed out (64 -> 33 -> stopped here).** Second
refinement pass reduced flagged rows from 64 to 33, but manual read showed
diminishing returns: remaining false positives are driven by real clinical
synonyms the word-overlap check can't see (BiPAP = NIV; "suprapubic
prostatectomy" = "open prostate surgery," caught correctly and identically
by all 4 combos on query 18; "ages 6 to 21" missed by the threshold-filter
regex, which only caught symbol ranges and singular "age"). Chose to stop
patching the regex here rather than chase every synonym pattern -- real
diminishing returns for a portfolio-project timeline, and worth stating as
a limitation rather than hiding it.

**Final tally for the taxonomy, this screen:** one confirmed real
hallucination (query 33/NCT01223885, pregnancy -- fabricated, not merely
absent from context), one confirmed criterion-scope misapplication (query
1/NCT00643591, spine treated as equivalent to head/neck/brain), one new
confirmed label/rationale self-contradiction (query 24/NCT02485808 --
rationale explicitly concludes gross hematuria is absent and no exclusion
applies, but the structured label says excluded anyway; more damning than
the example since it uses no hedging language at all, so it wouldn't
have been caught by `check_label_consistency.py`'s phrase-matching).
Everything else in the 33 is heuristic noise, not a real model error.

**A distinction worth keeping for future reference, not resolved today:**
the heuristic can't tell "honestly noting information is absent" (query
60/NCT03372096 -- model correctly states the vignette never mentions prior
BPH medication trials, a legitimate, non-fabricated observation) apart from
"fabricating a fact that's absent" (the pregnancy case). Both produce zero
word-overlap. The former is arguably a labeling/calibration question
(should this have been insufficient_information rather than excluded?)
rather than an error at all.

**Interview narrative on the limits of this project (refined after
discussion, not accepted as originally framed):**

The original framing -- "we built a hallucination detector, it found no
real hallucinations, this shows DS/MLE approaches hit a ceiling without
clinical SME involvement" -- is directionally right but imprecise in two
ways worth correcting before it goes in an interview:

1. *The screen did find real things.* One confirmed hallucination
   (pregnancy case), one confirmed criterion-scope misapplication, one
   confirmed label/rationale contradiction -- 3 real findings out of 33
   flagged (~9% precision). A working tool at low precision, not a null
   result. Worth saying so.

2. *Most of the false positives were an engineering gap, not a clinical
   one.* BiPAP=NIV, "suprapubic prostatectomy"=open prostate surgery --
   these are synonym-matching failures in a literal word-overlap check,
   fixable with better technical methods (semantic similarity, a medical
   ontology, or an LLM-based grounding judge) without needing a clinician
   in the loop at all. Framing this specific limitation as "needs an SME"
   invites an obvious interview counter: "couldn't embeddings have solved
   this?" -- and the honest answer is yes, likely.

The real, defensible version of the ceiling claim is the confident
disagreement set, not the hallucination screen: cases like
ischemic vs. structural mitral regurgitation, or "renal failure requiring
RRT" meaning chronic dialysis-dependence vs. incidental acute dialysis
during sepsis. No amount of better grounding-check tooling resolves those
-- they require actual judgment about disease mechanism and trial-design
intent. That's the genuine distinction worth making in interviews: telling
apart "this is an engineering problem, solvable with a better check" from
"this is a judgment problem, solvable only with domain expertise" -- and
being able to point to a concrete example of each from the same project.

## 5 — Cost/quality frontier

| combo | $/1k pairs | accuracy_strict | mean_clinical_cost | abstention_rate | missed_exclusion_rate |
|---|---|---|---|---|---|
| haiku_zero_shot | $2.60 | 0.558 | 0.550 | 0.183 | 0.050 |
| haiku_few_shot | $2.95 | 0.592 | 0.496 | 0.158 | 0.042 |
| sonnet_zero_shot | $6.55 | 0.592 | 0.283 | 0.250 | 0.000 |
| sonnet_few_shot | $7.57 | 0.558 | 0.304 | 0.275 | 0.000 |

**accuracy_strict is not a useful differentiator here** -- 0.558-0.592 across
all four, too narrow a band to distinguish anything. The real signal is in
mean_clinical_cost and missed_exclusion_rate: both Sonnet combos hit zero
missed exclusions across 120 pairs each; both Haiku combos miss 4-5% of
the time. Sonnet's clinical cost (~0.28-0.30) is roughly half Haiku's
(~0.50-0.55).

**Few-shot does not earn its cost for Sonnet.** `sonnet_zero_shot` beats
`sonnet_few_shot` on every single metric -- cheaper, same accuracy, better
clinical cost, less hedging, same missed-exclusion rate. Strictly
dominated; no reason to keep few-shot for Sonnet. For Haiku, few-shot
helps marginally (slightly better accuracy and clinical cost) but costs
more too -- a real but small tradeoff, not a clean win either way.

**Selected: `sonnet_zero_shot`.** Best clinical cost, tied-best accuracy,
zero missed exclusions, and dominates its own few-shot variant outright.
Costs 2.5x more than `haiku_zero_shot` per pair.

**Headline framing for the report, and the more interesting finding than
"which model wins":** at this scale the dollar difference is close to
irrelevant -- $2.60 vs $7.57 per *thousand* pairs is a rounding error for
almost any real deployment. This isn't really a cost-constrained frontier;
it's a safety/accuracy tradeoff where dollar cost barely factors in until
volume reaches into the millions of pairs. The decision that matters is
which errors you're willing to tolerate, not the budget -- which is
exactly why the clinically-weighted metric was worth building instead of
leaning on raw accuracy alone.

## 5 (cont.) — Held-out test, 2022, sonnet_zero_shot

| metric | dev (2021) | test (2022) |
|---|---|---|
| accuracy_strict | 0.592 | 0.442 |
| abstention_rate | 0.250 | 0.442 |
| mean_clinical_cost | 0.283 | 0.371 |
| missed_exclusion_rate | 0.000 | 0.008 (1/120) |
| recall_eligible | 0.333 | 0.200 |
| cost per 1k pairs | $6.55 | $6.52 |

**Correction on causality, worth getting right for the interview:** the
clinically-weighted cost metric does NOT influence the model's behavior --
it never sees those weights, they're computed after the fact purely for
scoring. Low recall_eligible is not "caused by" the cost weighting. The
real, model-side driver is the same structural one identified on the dev
set: asserting "eligible" requires positively confirming every criterion;
asserting "excluded" (or hedging) requires only finding/flagging one gap.
That's a property of the task and the prompt's own instructions, not of
how we chose to score the output afterward.

**Real generalization gap, reported honestly rather than smoothed over:**
abstention nearly doubled (0.250 -> 0.442) and accuracy_strict dropped
(0.592 -> 0.442) from dev to held-out test. Dev-set performance did not
fully hold on unseen data.

**What did generalize: the safety property.** missed_exclusion_rate stayed
essentially at zero (1/120) on held-out data, even as the model's
willingness to commit to a positive "eligible" call dropped sharply. The
caution generalized; the raw hit-rate didn't. Worth stating as the honest
headline: this config is safe out of sample, even though it's not highly
decisive out of sample.

**Open hypothesis, not yet confirmed:** 2022's topics are the cleaner,
more textbook-style vignettes (shorter, less like real
clinical notes) -- plausible this contributes to higher abstention, since
a terser vignette gives less to positively confirm eligibility against.
Worth checking if time allows; not chased further today.

**haiku_zero_shot held-out comparison (n=120, same 2022 test set):**

| metric | haiku (test) | sonnet (test) |
|---|---|---|
| accuracy_strict | 0.467 | 0.442 |
| abstention_rate | 0.300 | 0.442 |
| mean_clinical_cost | 0.650 | 0.371 |
| missed_exclusion_rate | 0.067 (8/120) | 0.008 (1/120) |
| cost per 1k pairs | $2.59 | $6.52 |

**Decisive result: the 8x gap in missed_exclusion_rate, and it's a real
generalization result, not a dev-set fluke.** Haiku's dangerous-error rate
rose slightly from dev (0.050) to test (0.067); Sonnet's stayed pinned
near zero (0.000 -> 0.008). This is the one number in the project that
actually tests whether Sonnet's safety advantage was real or an artifact
of the 2021 sample it was tuned on -- it held up.

**Raw accuracy is a wash (0.442 vs 0.467) -- if that were the only metric
examined, Haiku would look like the free win with no real cost.** That's
the whole reason the clinically-weighted metric was worth building instead
of leaning on accuracy alone; it's the only one of the five metrics that
actually differentiates the two configs correctly here.

**The real cost of choosing Sonnet is not the ~$4/1k-pairs dollar
difference (trivial, as established) -- it's the higher abstention rate:
44.2% vs 30.0%.** That's an honest operational tradeoff (more cases
requiring human review), not something to gloss over. Sonnet isn't
strictly better than Haiku; it's safer at the cost of being less decisive.

## Final decision: production config locked as `sonnet_zero_shot`

Selected on the strength of the held-out missed-exclusion result (8x safer
than Haiku, generalizes cleanly out of sample) and the clinically-weighted
cost metric (0.371 vs 0.650), with the abstention-rate tradeoff (44% of
eligible cases routed to human review) stated explicitly as the real price
of that choice, not hidden. Dollar cost was not the deciding factor at any
point in this decision -- both configs are cheap enough in absolute terms
that cost was never the binding constraint; the choice was entirely about
which error profile is acceptable for a clinical-adjacent use case.

## 6 — Flask UI

Three views (dashboard, case explorer, live demo), model selection
centralized in `app/config.py` as a single source of truth read by both
backend routing and UI copy -- each model's `tradeoff_*` fields are real
held-out numbers, not marketing copy. Deployed app deliberately lightweight
(`flask`, `requests`, `anthropic` only) -- all heavy ML deps (`torch`,
`sentence-transformers`, `pyarrow`) stay in the pipeline's requirements,
never ship to the deployed app, since `data_prep.py` bundles everything
into a static JSON ahead of time.

One real bug found and fixed during smoke-testing: `_ensure_live_deps()`
sat outside the try/except in `/api/live_judge`, so a missing dependency
or API key produced a raw unhandled 500 instead of a clean JSON error.
Confirmed via test client before and after the fix.

Live-tested by the user against real patient input: live ClinicalTrials.gov
search + live judge call worked end-to-end, no errors, multiple trials
tried. Explorer-page UI organization flagged as a follow-up, not urgent.

## 7 — Deploy config + README

Added `app/requirements.txt` (deploy-only, lightweight), `Procfile` and
`render.yaml` for deployment, updated `app.py` to respect the `PORT` env
var and disable debug mode by default outside local dev. Verified the
actual deploy command (`gunicorn app:app` from the `app/` working
directory) serves both routes correctly before shipping it as the
documented config, not just assumed.

Rewrote the top-level README to lead with the headline finding (missed-
exclusion rate, 8x gap, held up out of sample) rather than the pipeline
architecture, per the original plan's own stated goal -- a general
reviewer should get the so-what in the first paragraph, not after wading
through setup instructions.

## Post-launch — eval rigor additions (Tier 1-2)

Prompted by explicitly targeting an LLM-evaluation-titled role: raw
accuracy and a single held-out run aren't enough rigor for that audience
specifically. Added three checks the original plan didn't include.

**Tier 1a: statistical significance on the headline claim.** The
missed-exclusion-rate gap (1/120 vs 8/120, reported as "8x") needed a
significance test, not just a ratio. Methodological correction made along
the way: the correct at-risk denominator is the 60 ground-truth-excluded
pairs specifically, not all 120 -- the 60 eligible-ground-truth pairs are
structurally incapable of producing a missed exclusion, so including them
dilutes the proportion test. On the correct denominator (`significance_test.py`,
Fisher's exact test): p=0.032 (two-sided), already significant at n=60/
model. Wilson 95% CI: Sonnet 1.7% [0.3%, 8.9%], Haiku 13.3% [6.9%, 24.2%]
-- CIs still overlap despite significance, meaning the direction is real
but the precision is genuinely loose at this sample size. Honest
conclusion: the finding is real, not yet tightly pinned down.

**Tier 1b: scaling the held-out sample.** n_per_class raised from 60 to
200 (400 total pairs) to tighten the CI above -- cost estimated at ~$3.65
total for both models, trivial relative to the value of a defensible
confidence interval on the project's headline claim.
*(re-run results: TODO once executed)*

**Tier 2a: order-sensitivity check.** Known LLM-judge failure mode from
the eval literature -- does shuffling the order of a trial's criteria
change the verdict for unchanged underlying facts? `order_sensitivity_test.py`
runs each sampled pair twice (original order, independently-shuffled
inclusion/exclusion lists) and reports the flip rate. Deliberately
excludes pairs where neither list has >=2 items, since those can't
actually be reordered and would dilute the rate with untested cases.
*(results: TODO once executed)*

**Tier 2b: self-consistency check.** Distinct from accuracy -- does the
judge give the same answer to the identical question asked multiple
times? `self_consistency_test.py` calls the judge k=3 times per sampled
pair on identical input and reports the unanimous-verdict rate. Matters
operationally: an unstable judge means the same patient-trial pair could
get a different answer depending on when you ask, independent of whether
either answer is correct.
*(results: TODO once executed)*

**Tier 3 (selective-prediction/risk-coverage analysis, LLM-judge-as-tagger
validation, second-vendor model comparison) scoped but deferred** --
documented as explicit future work rather than built, given diminishing
returns against the portfolio timeline. A clear answer to "what would you
add with more time" is close to as valuable as building it.

## Tier 1b results: n=500/class held-out re-run — the headline claim revised

Re-ran the held-out test at n=500/class (1,000 pairs total per model, up
from 120) to tighten the CI on the missed-exclusion gap. Result: **the
finding is real but was overstated at the smaller sample.**

| metric | n=60/class (old) | n=500/class (new) |
|---|---|---|
| Sonnet missed-exclusion (at-risk: excluded-gt only) | 1/60 = 1.7% | 13/500 = 2.6% |
| Haiku missed-exclusion (at-risk) | 8/60 = 13.3% | 26/500 = 5.2% |
| Ratio | ~8x | ~2x |
| Fisher's exact p-value (two-sided) | 0.032 | 0.049 (barely significant) |
| Sonnet 95% Wilson CI | [0.3%, 8.9%] | [1.5%, 4.4%] |
| Haiku 95% Wilson CI | [6.9%, 24.2%] | [3.6%, 7.5%] |
| Abstention rate (Sonnet / Haiku) | 44.2% / 30.0% | 39.7% / 31.3% |
| Abstention gap | 14.2 points | 8.4 points |
| accuracy_strict (Sonnet / Haiku) | 44.2% / 46.7% | 49.0% / 53.7% |
| mean_clinical_cost (Sonnet / Haiku) | 0.371 / 0.650 | 0.364 / 0.411 |

**What actually happened, precisely:** Haiku's small-sample
missed-exclusion estimate (13.3%) was the unstable part, not Sonnet's.
At n=500 Haiku's rate more than halved to 5.2%, while Sonnet's held
roughly steady (1.7% -> 2.6%). The "8x" headline was mostly an artifact
of Haiku drawing an unlucky small sample, not Sonnet being exceptionally
safe. The true effect appears to be roughly 2x, and it remains
statistically significant (p=0.049) -- but "barely significant, CIs
still overlapping" is the honest characterization, not "clear-cut."

**Two findings that did NOT get revised down, worth contrasting
directly against the one that did:**
- Ground-truth-ambiguity rate: 46/120 (38.3%) at the small sample,
  414/1,000 (41.4%) at the large one -- stable, not a small-sample
  artifact. Pattern composition (consensus_hedge-dominant) held too.
- Sonnet's clinical-cost advantage over Haiku persists (0.364 vs 0.411),
  though the *margin* narrowed substantially (43% relative gap ->
  11.5%), tracking Haiku's broad improvement with more data, not just
  its missed-exclusion rate specifically.

**New finding at this scale, arguably a better illustration of the
project's core argument than the original numbers were:** raw
accuracy_strict now favors Haiku (53.7% vs Sonnet's 49.0%) -- a real gap
in the *opposite* direction from the clinically-weighted metric. At n=60
the two models were merely statistically tied on accuracy; at n=500,
accuracy alone would actively point you at the wrong model for the
reason that actually matters.

**Decision: Sonnet recommendation held, not reversed, on this basis.**
The clinical-cost metric and the (now-narrower, still-significant)
missed-exclusion gap both still favor Sonnet. But the human cost of that
choice is now the more material tradeoff to reason about explicitly, not
the API cost (which remains trivial, ~$4/1,000 pairs): Sonnet routes
8.4 percentage points more cases to human review than Haiku -- roughly
84 more reviews per 1,000 patient-trial pairs judged. Human reviewer
time almost certainly costs more per unit than LLM tokens, so this is
likely the more expensive line item between the two models even without
knowing the exact reviewer-capacity constraint -- though a real
deployment decision would want that number specifically, not an
assumption. Flagged explicitly as an open question the project doesn't
resolve, rather than quietly assumed away.

**Explicit, disclosed stopping point:** did not scale further than
n=500/class. The CIs still overlap and a larger sample would likely
tighten them further, but the cost of continuing to spend on API credits
for a portfolio project outweighed the value of a marginally tighter
interval on an already-significant result. Stated as a deliberate
tradeoff, not hidden as a limitation discovered later.

**Housekeeping:** the "8x" claim previously stated in the README headline
and `app/config.py`'s UI tradeoff copy has been corrected throughout to
reflect the n=500 numbers, so the repo doesn't contain two documents that
quietly disagree with each other.

