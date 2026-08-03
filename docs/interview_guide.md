# Interview Guide: Clinical Trial Matching Eval Harness

This is your study document, not something to memorize word-for-word. Read
it a few times, then talk through the sections out loud on your own before
the interview. Every concept below is explained using your own project's
real numbers and real examples — not generic textbook cases — because
that's what you'll actually be asked to walk through.

---

## Part 1: The 30-second pitch

Use this as your answer to "walk me through this project" or "tell me
about a project you're proud of."

> "I built an eval harness for LLM-based clinical trial matching — scoring
> a retrieve-then-judge pipeline against real physician eligibility
> decisions from a NIH/NLM benchmark called TREC Clinical Trials. The
> headline finding: on a 500-case held-out sample per model, one judge
> config misses a true trial exclusion about twice as often as another —
> a real, statistically significant gap, but a modest one. I actually
> found an 8x gap first, on a smaller 120-case sample, didn't trust it,
> ran a properly-powered follow-up, and the true effect turned out to be
> about a quarter the size — which is exactly the kind of thing a
> significance test exists to catch, and I think that revision is a
> better story than the original dramatic number would have been. I
> built the whole thing: real data pipeline, a retrieval comparison with
> a genuinely counterintuitive finding, a judge evaluated on the error
> that actually matters clinically rather than just accuracy, known
> LLM-judge bias checks, an error taxonomy built from real case-by-case
> adjudication, and a deployed app with a live demo anyone can try."

If they want more, go to whichever section below matches what they're
asking about.

---

## Part 2: Concepts you need cold

You said LLM eval and metrics like nDCG@10 are new to you. Here's every
concept you'll plausibly get asked about, explained with your own data.

### Embeddings and retrieval

**What an embedding is.** A way of turning text into a list of numbers
(a vector) such that texts with similar *meaning* end up as similar
numbers — "geometrically close." You compare two pieces of text not by
matching words, but by measuring how close their number-vectors are.

**Retrieval.** Given a patient description, you embed it, embed every
trial in your database, and pull back the trials whose vectors are
closest to the patient's vector. That's the "R" in RAG (retrieval-
augmented generation) — you retrieve relevant documents *before* asking
the LLM to reason, rather than relying on what the LLM already "knows."

**Why you compared two embedding models.** You tested a general-purpose
model (`all-mpnet-base-v2`, trained on broad web text) against a
purpose-built biomedical one (MedCPT, from NIH, trained on 255 million
real PubMed search-log query-to-article pairs). The intuitive guess is
"biomedical should win, it's the right domain." It didn't — general beat
it roughly 2x. **The finding, in one sentence you should have ready:**
*"Domain match on content isn't the same as domain match on register — MedCPT
had never seen textbook-style diagnostic vignettes as an input format,
only real researcher search queries, even though the subject matter was
nominally the same."*

### nDCG@10 — the metric that was new to you

Think of it like this: if you search something and the right answer is
result #1, that's great. If it's #10, that's technically "found" but much
less useful. Recall (see below) treats both cases as equally "correct" —
nDCG doesn't. **nDCG@10 measures ranking quality, not just whether the
right thing was found somewhere in the top 10 — it rewards putting the
best matches at the top, not just anywhere in the list.**

The "10" means it only looks at the top 10 results. In your project,
"right answer" wasn't binary — a trial could be "eligible" (best match,
counted double) or "excluded" (still relevant, half credit) or "not
relevant" (zero credit) — so nDCG also rewards ranking the *eligible*
trials above the merely *excluded-but-related* ones, not just finding
something in the right disease area.

**Your number:** general-purpose embeddings scored ~0.46-0.49 nDCG@10;
MedCPT scored ~0.22-0.27. If someone asks "is 0.49 good?" — your honest
answer: *"it lands in the range published TREC Clinical Trials track
systems actually achieved, which was my sanity check that the number
wasn't corrupted or unrealistically inflated."*

### Recall@k

Simpler than nDCG: **of all the trials that were actually relevant to
this patient, what fraction did you manage to pull into your top-k
results?** Recall@10 = 0.078 means: out of everything genuinely relevant
to a patient, you only caught about 8% of it in your top 10 guesses.
That sounds low, but remember the corpus has ~48,700 trials and a typical
topic has 100+ genuinely relevant ones — catching some of the best ones
early is the goal, not catching literally everything.

### Precision vs. recall (for the judge, not retrieval)

- **Recall** (for the "eligible" class): of all patients who *should* have
  been called eligible, what fraction did the judge actually call
  eligible? Yours was low (0.20-0.33) — the judge often hedged instead.
- **Precision** (for "eligible"): of everything the judge *did* call
  eligible, what fraction was actually correct? Yours was very high for
  Sonnet (0.92-1.00) — when it commits to "eligible," it's almost always
  right.
- **In one sentence:** *"The judge is conservative — it rarely says
  eligible when it shouldn't, but it also often declines to say eligible
  when it probably should have. High precision, lower recall."*

### Confusion matrix

A simple grid: rows are what's actually true (physician ground truth),
columns are what the model predicted. The diagonal is where you want
mass to pile up (correct). Off-diagonal cells are errors — and your
project's whole argument is that **not all off-diagonal cells cost the
same.** Predicting "eligible" when truth is "excluded" (the dangerous
cell) is the costly direction; predicting "excluded" when truth is
"eligible" is annoying but not dangerous.

### LLM-as-judge / structured output

Instead of asking the LLM to write free-text and then trying to parse
"eligible" or "excluded" out of prose (fragile, error-prone), you forced
it to respond through a structured tool call with three required fields:
`label`, `cited_criterion`, `rationale`. This guarantees a parseable
answer every time and — importantly — forces the model to name the
specific criterion driving its decision, which is what let you audit its
reasoning afterward rather than trust it blindly.

### Hallucination (in this context)

Not "the model made a math error" — specifically, **the model asserted a
fact about the patient that isn't anywhere in the source text.** Your
confirmed example: a trial's only exclusion was "pregnant women," the
patient vignette never mentions pregnancy at all, and the model cited
pregnancy as the reason for exclusion anyway. That's the difference
between an error and a hallucination — an error is being wrong about
something in the text; a hallucination is inventing something that isn't.

### Abstention rate

How often the judge said "I don't have enough information" instead of
committing to eligible/excluded. This isn't automatically bad — a system
that correctly declines to guess is safer than one that guesses wrong.
Your selected model abstains 39.7% of the time on held-out data. That's a
real, honest cost (more cases need a human reviewer), not something to
hide.

### The clinically-weighted cost metric

You built a custom scoring rule reflecting that **not all errors are
equally bad**: a missed exclusion (telling someone they're eligible when
they're not) costs 5 points; an over-exclusion (telling someone they're
not eligible when they are) costs 1 point; a hedge costs 0.5; correct
costs 0. This is *not* a standard, textbook metric — it's a deliberate
judgment call you made and documented, reflecting the real-world stakes
of the task. **Be ready to defend the specific weights if asked** — your
honest answer is "these are illustrative, not derived from a formal cost
model; the point is demonstrating that error type matters, not that 5:1
is the precisely correct ratio."

### Dev set / held-out test set

You tuned everything — model choice, prompt strategy — using only the
2021 data (the "dev set"). The 2022 data (the "held-out test set") was
never touched until the config was already locked. This matters because
it's the only way to know if your results generalize or if you just got
lucky/overfit to quirks of one year's data. **Your evidence this
discipline paid off:** Sonnet's safety advantage over Haiku held in
*direction* from dev-era comparisons through to the held-out 2022 test —
it never flipped. Note this is a different check from the sample-size
story above: the *direction* generalized across years (dev discipline
working as intended); the *magnitude* (8x vs. ~2x) needed a separate,
later correction from running a bigger sample within the same 2022 test
year. Two different validations, worth keeping distinct if asked.

### Statistical significance, p-values, and confidence intervals

This became central to the project after the headline number changed
under scrutiny, so it's worth having cold.

**The problem a significance test solves.** You observed Sonnet missing
1/60 excluded cases and Haiku missing 8/60 — an "8x" difference. But with
counts that small, a natural question is: *could that just be luck?* If
you flipped a coin 60 times, you wouldn't be shocked to see 1 head one
time and a different small number another time, purely by chance. A
significance test asks: assuming there's actually no real difference
between the two models, how likely is it we'd see a gap this large just
by random sampling variation? If that probability (the **p-value**) is
low, you have evidence the gap is real, not noise.

**Your numbers:** Fisher's exact test (the right test for small-count
2x2 comparisons like this) gave p=0.032 at n=60/class, and p=0.049 at
n=500/class — both under the conventional 0.05 threshold, so both
"statistically significant." **Important nuance to have ready:** 0.049 is
*barely* under 0.05 — this is not a strong, decisive result, it's a
result that clears the bar narrowly. Say that plainly if asked; don't
oversell it.

**Confidence interval, in one sentence:** a range of plausible true
values for a measurement, given the sample size you have. A 95% CI means
"if I repeated this sampling process many times, about 95% of the
intervals I'd compute would contain the true value." Your Sonnet interval
at n=500 was [1.5%, 4.4%] — tighter than at n=60 ([0.3%, 8.9%]), because
more data narrows your uncertainty about the true rate.

**The critical judgment call you made and should be ready to explain:**
choosing the right denominator. A missed exclusion can only happen among
patients who were *truly* excluded — someone who was truly eligible can't
produce that specific error by definition. So the correct base rate is
"missed exclusions divided by truly-excluded cases," not "divided by all
cases." Using the wrong denominator would have diluted the signal with
cases that structurally couldn't produce the error being measured. *"I
explain that distinction because a good interviewer might probe exactly
there — testing whether you actually understand what you're measuring,
not just running a library function."*

**Why you didn't scale the sample even further.** The CIs still overlap
at n=500. A bigger sample would likely tighten them further. You made a
deliberate, disclosed choice to stop — a real statistical test costs real
API money, and the marginal value of a tighter interval on an
already-significant result was lower than the cost of continuing to pay
for it on a portfolio project. Say this plainly if asked, framed as a
resource tradeoff you made consciously, not a limitation you didn't
notice.

### Zero-shot vs. few-shot

Zero-shot: just instructions, no examples. Few-shot: instructions plus a
couple of worked examples in the prompt. The generic assumption is
"few-shot should help." In your project it didn't for the model you
selected — `sonnet_zero_shot` beat `sonnet_few_shot` on every single
metric. Worth having ready: *"I didn't assume more prompt engineering is
always better — I tested it and let the data override the intuition."*

---

## Part 3: The story, day by day (condensed narrative)

Use this if asked "walk me through your process" in more depth than the
30-second pitch.

**Data (Day 1).** Real ClinicalTrials.gov trials (~48,700), real physician
eligibility judgments from a NIH-affiliated benchmark (TREC), zero
synthetic data on the trial side. The one synthetic piece — patient
vignettes — was TREC's own design choice, not mine, because real patient
records can't legally be public. I explicitly chose this dataset over
alternatives (like MIMIC) partly because it doesn't require a data use
agreement, which mattered since this would be a public portfolio piece.

**Retrieval (Day 2).** Compared general-purpose vs. biomedical embeddings.
Along the way I caught and fixed three real bugs: a sampling bug where my
"random" 100-trial test was actually the *oldest* 100 trials (an
artifact of alphabetically-sorted trial IDs being roughly chronological);
a data-type bug where missing values were silently crashing the pipeline
because Python treats `NaN` as truthy; and — the big one — the biomedical
model's query encoder was silently truncating patient descriptions to 64
tokens when the real descriptions averaged 130-185 tokens, discarding
most of the clinical detail before the model ever saw it. Fixing the
truncation bug helped, but general-purpose embeddings still won —
because the biomedical model had never seen textbook-style vignettes in
training, only real research search queries.

**Judge (Day 3).** Built a grounded LLM judge with forced structured
output. Ran small manual spot-checks before scaling up, and found real,
specific error patterns early: the same missing-lab-value hedge pattern
appearing twice in 15 samples, a self-contradictory output where the
model's own stated reasoning didn't match its answer, and a case where a
cheaper model confidently gave the dangerous wrong answer while the more
capable model correctly hedged instead.

**Full eval harness (Day 4).** Scored all four model/prompt combinations
against ground truth. The critical realization: raw accuracy made two
very different models look identical, while a purpose-built clinical-cost
metric revealed one of them was making the dangerous error twice as
often. Built an error taxonomy from real case-by-case reading, not
keyword guessing — found a confirmed hallucination, a criterion
misapplication, and a label/reasoning contradiction. Also caught and
fixed a real methodological mistake in my own tooling: my first
hallucination-detector flagged 64 cases, and reading them showed almost
all were false positives from a fixable bug, not real hallucinations.

**Cost/quality frontier + held-out test (Day 5).** Locked in the final
model/prompt choice using real token costs, then ran it — for the first
time — against the 2022 data that had never been touched. The safety
advantage held up; the raw accuracy did not fully hold up, and I reported
both honestly rather than only the flattering number.

**UI + deployment (Days 6-7).** Built a three-page Flask app: a
precomputed report, a case browser with real examples (including the
hallucination case, shown deliberately, not hidden), and a live demo
against real currently-recruiting trials. Deployed with GitHub Actions —
static report on GitHub Pages, live backend on Render, split because only
one of them can legally be static (the live demo needs a secret API key
server-side, which static hosting can't provide).

**Eval rigor, post-launch (added specifically for this role).** Realized
raw accuracy and a single 120-pair held-out run weren't enough rigor for
an LLM-evaluation-titled role to find credible. Added a proper
significance test on the headline missed-exclusion claim, which
initially looked like an 8x gap — then, on a much larger 500-pair-per-
class re-run, revised that down to a real but more modest ~2x gap,
significant at p=0.049. Documented the revision as a finding, not a
mistake to hide. Also reframed the model-selection tradeoff away from
API dollar cost (trivial at this scale) and toward human review capacity
(abstention rate), since that's very likely the more expensive line item
in a real deployment.

---

## Part 4: Numbers to know cold

| What | Number |
|---|---|
| Trial corpus size | ~48,700 real ClinicalTrials.gov trials |
| Dev set (2021) | 75 patient topics, physician-judged |
| Held-out test (2022), final run | 500 pairs per class per model (1,000 total per model) |
| General vs. biomedical embeddings (nDCG@10) | ~0.46-0.49 vs. ~0.22-0.27 |
| Selected judge config | Claude Sonnet, zero-shot |
| Missed-exclusion rate, held-out, at-risk denominator (Sonnet) | 2.6% (13/500), 95% CI [1.5%, 4.4%] |
| Missed-exclusion rate, held-out, at-risk denominator (Haiku) | 5.2% (26/500), 95% CI [3.6%, 7.5%] |
| Significance of the gap | Fisher's exact, p=0.049 (two-sided) — real, but barely under 0.05 |
| Abstention rate, held-out (Sonnet / Haiku) | 39.7% / 31.3% — an 8.4-point gap |
| Accuracy_strict, held-out (Sonnet / Haiku) | 49.0% / 53.7% — favors Haiku |
| Clinical cost, held-out (Sonnet / Haiku) | 0.364 / 0.411 — still favors Sonnet, margin narrowed from 43% to 11.5% |
| Cost, Sonnet vs. Haiku | ~$6.34 vs. ~$2.52 per 1,000 pairs — not the deciding factor |

**The one number to lead with if you only remember one thing:** the
missed-exclusion gap is real and significant (p=0.049) but modest (~2x,
not 8x) — and the fact that you found the bigger number first, tested it
properly, and revised it down is itself the strongest single thing to
say about your eval rigor.

---

## Part 5: Anticipated questions and how to answer them

**"Walk me through this project."**
Use the 30-second pitch from Part 1. Let them steer you deeper from there.

**"Why did you pick this dataset?"**
Real data was a hard requirement. TREC Clinical Trials was the only
option that was simultaneously public/no-DUA, had physician-adjudicated
ground truth already built for exactly this task, and was queryable
immediately — versus alternatives like MIMIC or n2c2 which needed a data
use agreement that would have blocked the timeline.

**"What is nDCG and why did you use it?"**
Give the ranking-quality explanation from Part 2. Add: *"I used it
instead of just accuracy because retrieval isn't binary found/not-found —
where in the ranking something appears matters, and nDCG is the standard
way to score that."*

**"Why did the biomedical embedding model lose?"**
*"It was trained on real PubMed search queries — short, terse research
questions — not on textbook-style clinical vignettes. When I read through
the trials it failed on hardest, they clustered specifically around that
clean, templated writing style, not around real messy clinical notes,
which it handled fine. So the lesson wasn't 'domain-specific models are
worse' — it was that domain match on subject matter isn't the same as
domain match on the actual writing style of your input."*

**"How did you evaluate the judge?"**
Explain the three-class problem (eligible/excluded/insufficient
information vs. two-class ground truth), the confusion matrix, and pivot
quickly to the clinically-weighted cost metric as your actual headline
metric, not accuracy.

**"Why Sonnet over the cheaper model?"**
Lead with the honest, current numbers: *"Sonnet misses a true exclusion
in about 2.6% of at-risk cases versus Haiku's 5.2% — roughly twice as
often, and that gap is statistically significant, though not by a wide
margin. Raw accuracy actually favors Haiku, which is exactly why I don't
lead with accuracy — it would point you at the wrong model for the
reason that matters. The real cost of choosing Sonnet isn't the API
price difference, which is trivial — it's that Sonnet routes about 8
more cases per 100 to human review instead of committing to an answer.
I chose the safer error profile knowingly, with that cost stated
explicitly, not for free."*

**"Didn't you say it was an 8x gap earlier? What happened?"**
This is a great question to want asked, not fear: *"That was my first
estimate, at 60 pairs per class. I didn't trust a conclusion built on
that few missed-exclusion events, so I reran it at 500 pairs per class
and tested it properly with Fisher's exact test. The true gap turned out
to be about a quarter the size — real and significant, but not dramatic.
The 8x number was mostly Haiku drawing an unlucky small sample, not
Sonnet being unusually safe. I'd rather tell you about a claim I revised
under scrutiny than one I never checked."*

**"How constrained is the human review capacity this depends on?"**
Be honest about the gap: *"I don't have a real number for that, and I
say so explicitly rather than assume one. The point I can defend is the
relative one — reviewer time almost certainly costs more per case than
an LLM API call, so the abstention-rate difference is probably the more
expensive line item between these two models even without knowing the
exact capacity constraint. But a real deployment decision would need
that number specifically, and I've flagged it as an open question this
project doesn't resolve, not something I quietly assumed."*

**"What was your biggest finding?"**
Two good answers depending on what they want:
- Technical: the embedding genre-mismatch finding (general beat
  biomedical because of input register, not subject domain).
- Eval-craft: that raw accuracy hid a real, important difference between
  models that only a task-specific weighted metric revealed.

**"Tell me about a mistake or bug you found."**
Pick whichever fits the conversation — you have several real, good ones:
the sampling-order confound (see Part 3), the MedCPT truncation bug, or
the hallucination-detector's own false-positive problem. All three follow
the same shape: *found it, diagnosed the root cause, fixed it, verified
the fix* — that shape is what you want to demonstrate, not just "I found
a bug."

**"You're not a clinician — how did you handle the clinical reasoning?"**
Be honest and specific: *"For most of the error analysis, the reasoning
is checkable without deep clinical training — does the cited criterion
actually appear in the trial text, does the patient's stated fact match
or contradict it. For the genuinely ambiguous clinical cases — like
whether a mitral regurgitation etiology counts as 'structural' versus
'ischemic' — I used a cross-model agreement method: when multiple
independent models agreed with each other but disagreed with the
physician judgment on the same case, that's evidence the physician likely
had chart context the vignette didn't capture, not evidence the models
were wrong. That let me separate 'real error' from 'likely ground-truth
ambiguity' without needing to personally adjudicate contested clinical
distinctions."*

**"What would a real clinical deployment need that this doesn't have?"**
Be ready with this list, stated plainly: SME/clinical validation of the
error taxonomy at scale (not just the initial adjudication I did),
broader model coverage (I tested 2 models, not a full sweep), real data
on human reviewer capacity and cost to properly weigh the abstention-rate
tradeoff instead of reasoning about it qualitatively, rate limiting on
the live endpoints, and likely a hybrid retrieval approach (reranking,
not just single-pass embedding search) for production-grade recall.

**"How do you know your own eval tooling is trustworthy?"**
This is your best "evaluate the evaluator" story: *"My first hallucination
detector flagged 64 cases. I read a sample before trusting the number and
found almost all were false positives — the heuristic couldn't handle
threshold comparisons like age cutoffs, where the whole point is that the
cited number and the patient's number are supposed to differ. I fixed
that, re-ran it, got 33, read those too, and found the remaining false
positives were driven by real clinical synonyms my literal text-matching
couldn't see — like BiPAP being the same thing as non-invasive
ventilation. I stopped patching there because I'd hit real diminishing
returns, and documented that limitation explicitly rather than pretending
the tool was more precise than it was."*

**"What's the difference between an engineering problem and a domain-
expertise problem in this project — can you give an example of each?"**
This is a great question to have a sharp answer ready for, since it shows
judgment, not just execution:
*"The hallucination-detector's false positives were an engineering
problem — literal word-matching couldn't see synonyms, and that's
solvable with a better technical method, like semantic similarity or an
LLM-based grounding check, no clinician required. The genuinely hard
cases were different — whether a specific mitral regurgitation mechanism
counts as 'structural' versus the trial's required 'ischemic' etiology.
No amount of better text-matching resolves that; it requires actual
clinical judgment about disease mechanism. Being able to tell those two
apart, instead of assuming every unresolved case needs a clinician, is
itself part of good eval design."*

---

## Part 6: Honest limitations — be ready before they ask

A good interviewer will probe for these. Naming them yourself, unprompted
if it fits naturally, reads as more credible than waiting to be caught.

- **Sample size, addressed but not fully closed.** Started at 120 pairs,
  found the CIs too wide to trust, scaled to 500 pairs per class. The
  missed-exclusion gap is now statistically significant, but the CIs
  still overlap — the direction is established, the precise magnitude
  isn't fully pinned down. Chose not to scale further, as a deliberate,
  disclosed tradeoff against continued API spend on a portfolio project,
  not an oversight.
- **Two models only.** Sonnet and Haiku, both from one vendor. No GPT/
  Gemini comparison, which would strengthen the "does this generalize
  across model families" claim.
- **Synthetic patient vignettes.** Not real patient charts — TREC's own
  design choice, but it means some ground-truth disagreements may reflect
  chart context no vignette-based system could ever see.
- **Retrieval is single-pass embedding search**, no reranking or hybrid
  lexical+semantic approach — a real production system would likely want
  both.
- **The hallucination screen is a heuristic**, not a validated detector —
  documented false-positive/negative behavior, not a tool you'd trust
  unsupervised.
- **Live demo has no rate limiting yet** on endpoints that call paid,
  external APIs — a known, flagged gap before wider sharing.

---

## Part 7: If asked to explain something on a whiteboard

**"Explain retrieve-then-judge in 30 seconds."**
Two stages. Retrieval finds candidate trials for a patient using
embedding similarity — fast, cheap, approximate. The judge then reasons
carefully over just the retrieved candidates, comparing the patient's
specific stated facts against the trial's actual criteria text, and
returns a structured verdict with a citation. Splitting it this way means
you can evaluate and improve each stage independently — which is exactly
what happened here, since the retrieval fix (Day 2) and the judge fix
(Day 5) were separate, unrelated improvements.

**"Explain why grounding matters."**
Without grounding, an LLM asked "is this patient eligible for trial X"
might answer from its general training knowledge about what trials
"usually" require — which can be wrong, outdated, or just invented. By
forcing the judge to see and cite the trial's *actual* current criteria
text, you eliminate that failure mode structurally, not just hope the
model behaves.
