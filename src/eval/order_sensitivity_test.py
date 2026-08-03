"""
Order-sensitivity test: a known failure mode in the LLM-as-judge
literature is that judgments can be sensitive to the order items are
presented in, independent of their actual content.

Originally designed to shuffle individual items within the inclusion/
exclusion lists. Revised after finding that a large fraction of trials in
the held-out sample store criteria as a single unbulleted paragraph (the
Day 1 eligibility parser only splits on bullet markers, correctly and
conservatively refusing to guess where to split a comma-separated
sentence) -- meaning many pairs have nothing to shuffle at the item
level. Switched to a more standard version of this test instead: does
swapping which BLOCK comes first in the prompt -- inclusion criteria
before exclusion, vs. exclusion before inclusion -- change the verdict
for the same underlying facts? This works regardless of how many items
are in either list, and is arguably the more commonly cited version of
order sensitivity in the eval literature anyway (prompt-structure order
effects, not just within-list ordering).

    python src/eval/order_sensitivity_test.py --n_sample 40
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judge"))
from prompts import build_prompt, JUDGMENT_TOOL
from judge_client import MODELS as MODEL_IDS, client as anthropic_client

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "order_sensitivity_results.jsonl"


def parse_list_col(x):
    """See src/judge/sample_pairs.py's comment: criteria columns are
    JSON-encoded on write, must be json.loads'd on read."""
    if isinstance(x, list):
        return x
    if pd.isna(x) or x == "":
        return []
    return json.loads(x)


def judge_with_block_order(patient_text, trial_title, inclusion, exclusion,
                            model_key, strategy, exclusion_first=False, max_retries=3):
    """Same as judge_client.judge_pair, but optionally presents the
    exclusion-criteria block before the inclusion-criteria block in the
    user message, to test prompt-structure order sensitivity."""
    import time

    incl_text = "\n".join(f"- {c}" for c in inclusion) if inclusion else "(none stated)"
    excl_text = "\n".join(f"- {c}" for c in exclusion) if exclusion else "(none stated)"

    system, _ = build_prompt(patient_text, trial_title, inclusion, exclusion, strategy)

    if exclusion_first:
        user_message = f"""Patient vignette:
{patient_text}

Trial: {trial_title}

Exclusion criteria:
{excl_text}

Inclusion criteria:
{incl_text}

Determine eligibility for this specific patient."""
    else:
        user_message = f"""Patient vignette:
{patient_text}

Trial: {trial_title}

Inclusion criteria:
{incl_text}

Exclusion criteria:
{excl_text}

Determine eligibility for this specific patient."""

    model = MODEL_IDS[model_key]
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = anthropic_client.messages.create(
                model=model, max_tokens=500, system=system,
                messages=[{"role": "user", "content": user_message}],
                tools=[JUDGMENT_TOOL],
                tool_choice={"type": "tool", "name": "submit_eligibility_judgment"},
            )
            tool_use_block = next(b for b in resp.content if b.type == "tool_use")
            return tool_use_block.input
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Judge call failed after {max_retries} attempts: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_path", default=str(PROCESSED_DIR / "judge_test_sample_2022.csv"))
    ap.add_argument("--n_sample", type=int, default=40)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--strategy", default="zero_shot")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_csv(args.sample_path)
    df["inclusion_criteria"] = df["inclusion_criteria"].apply(parse_list_col)
    df["exclusion_criteria"] = df["exclusion_criteria"].apply(parse_list_col)
    sample = df.sample(n=min(args.n_sample, len(df)), random_state=args.seed)

    print(f"Testing block-order sensitivity on {len(sample)} pairs "
          f"({args.model}/{args.strategy}, 2 calls each = {len(sample)*2} total calls)")

    results = []
    flips = 0

    for row in tqdm(sample.itertuples(), total=len(sample)):
        original = judge_with_block_order(
            row.patient_text, row.brief_title, row.inclusion_criteria,
            row.exclusion_criteria, args.model, args.strategy, exclusion_first=False,
        )
        swapped = judge_with_block_order(
            row.patient_text, row.brief_title, row.inclusion_criteria,
            row.exclusion_criteria, args.model, args.strategy, exclusion_first=True,
        )

        flipped = original["label"] != swapped["label"]
        flips += flipped
        results.append({
            "query_id": row.query_id, "doc_id": row.doc_id,
            "original_label": original["label"], "swapped_label": swapped["label"],
            "flipped": flipped,
        })

    with open(OUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    rate = flips / len(sample) if len(sample) else float("nan")
    print(f"\nBlock-order flip rate: {flips}/{len(sample)} = {rate:.1%}")
    print(f"Results saved -> {OUT_PATH}")
    if flips:
        print("\nFlipped cases (verdict changed purely from swapping which "
              "criteria block came first, same underlying facts):")
        for r in results:
            if r["flipped"]:
                print(f"  query={r['query_id']} doc={r['doc_id']}: "
                      f"{r['original_label']} -> {r['swapped_label']}")


if __name__ == "__main__":
    main()
