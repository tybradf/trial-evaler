"""
Judge client: wraps the Anthropic API, forces structured output via tool
use (so we're never regex-parsing prose out of a hedging response), and
logs token usage per call so Day 5's cost/quality frontier doesn't require
re-running everything.

Requires ANTHROPIC_API_KEY in the environment (the SDK reads it automatically).

Model strings current as of this build -- verify against your account/docs
if these have changed by the time you run this:
    "claude-haiku-4-5-20251001"  -- cheap tier
    "claude-sonnet-5"             -- capable tier
"""
import time

import anthropic

from prompts import build_prompt, JUDGMENT_TOOL

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-5",
}

client = anthropic.Anthropic()


def judge_pair(patient_text: str, trial_title: str, inclusion: list[str],
               exclusion: list[str], model_key: str = "haiku",
               strategy: str = "zero_shot", max_retries: int = 3) -> dict:
    """Returns a dict with label, cited_criterion, rationale, plus token
    counts and latency for cost tracking. Raises on repeated failure rather
    than silently returning a default -- a silent failure would look like a
    (wrong) judgment in the eval, not a missing one."""
    system, user_message = build_prompt(patient_text, trial_title, inclusion, exclusion, strategy)
    model = MODELS[model_key]

    last_err = None
    for attempt in range(max_retries):
        try:
            start = time.time()
            resp = client.messages.create(
                model=model,
                max_tokens=500,
                system=system,
                messages=[{"role": "user", "content": user_message}],
                tools=[JUDGMENT_TOOL],
                tool_choice={"type": "tool", "name": "submit_eligibility_judgment"},
            )
            latency = time.time() - start

            tool_use_block = next(b for b in resp.content if b.type == "tool_use")
            result = tool_use_block.input

            return {
                "label": result["label"],
                "cited_criterion": result["cited_criterion"],
                "rationale": result["rationale"],
                "model": model,
                "strategy": strategy,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "latency_sec": round(latency, 3),
            }
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Judge call failed after {max_retries} attempts: {last_err}")
