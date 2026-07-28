"""
Prompt construction for the grounded eligibility judge. Two strategies:
zero-shot and few-shot. Few-shot examples are hand-authored, NOT sampled
from the qrels/dev set -- using real dev-set pairs as in-context examples
would leak eval signal into the prompt itself.

The judge is always grounded: it only ever sees the specific trial's own
retrieved criteria text, never asked to recall trial content from memory.
This is the core "R" in RAG for this project -- generation is conditioned
on retrieved text, not on the model's parametric knowledge of trials.
"""

SYSTEM_PROMPT = """You are assisting a clinical trial matching system. Given a patient's clinical vignette and ONE candidate trial's actual eligibility criteria, determine whether this specific patient would be ELIGIBLE, EXCLUDED, or if there is INSUFFICIENT_INFORMATION in the vignette to decide.

Rules:
- Base your judgment ONLY on the criteria text provided below. Do not use outside knowledge about the trial, and do not assume criteria that aren't stated.
- ELIGIBLE: the patient's stated characteristics satisfy the inclusion criteria and do not trigger any exclusion criterion.
- EXCLUDED: the patient's stated characteristics fail an inclusion criterion, OR trigger an exclusion criterion.
- INSUFFICIENT_INFORMATION: the vignette genuinely does not contain enough information to determine eligibility on a criterion that would otherwise be decisive (e.g. a specific lab value or prior treatment history is required but never mentioned). Do not use this as a default -- only when a specific, named criterion cannot be evaluated from the vignette.
- Pay close attention to negation ("no history of X" is different from "history of X"), time windows ("within the last 6 months"), and numeric thresholds -- these are the most common sources of error.
- You must cite the single specific criterion (quoted or closely paraphrased) that most influenced your judgment.
"""

FEW_SHOT_EXAMPLES = """
Example 1:
Patient vignette: "45-year-old woman with newly diagnosed stage II breast cancer, ECOG performance status 1, no prior chemotherapy or radiation. Normal renal and hepatic function."
Trial criteria:
Inclusion: Age 18 and over; ECOG performance status 0-2; histologically confirmed breast cancer
Exclusion: Prior chemotherapy or radiation therapy for this diagnosis; pregnant or nursing
Judgment: ELIGIBLE
Cited criterion: "Prior chemotherapy or radiation therapy for this diagnosis" (exclusion) -- patient explicitly has none, and meets ECOG and diagnosis inclusion criteria.

Example 2:
Patient vignette: "68-year-old man with type 2 diabetes on metformin, HbA1c 8.2%. No known cardiovascular disease."
Trial criteria:
Inclusion: Adults with type 2 diabetes, HbA1c between 7.0% and 9.0%
Exclusion: eGFR less than 45 mL/min/1.73m2; history of diabetic ketoacidosis
Judgment: INSUFFICIENT_INFORMATION
Cited criterion: "eGFR less than 45 mL/min/1.73m2" (exclusion) -- patient meets the inclusion criteria, but renal function (eGFR) is never mentioned in the vignette and is required to rule out this exclusion criterion.
"""


def build_prompt(patient_text: str, trial_title: str, inclusion: list[str],
                  exclusion: list[str], strategy: str = "zero_shot") -> tuple[str, str]:
    """Returns (system_prompt, user_message)."""
    incl_text = "\n".join(f"- {c}" for c in inclusion) if inclusion else "(none stated)"
    excl_text = "\n".join(f"- {c}" for c in exclusion) if exclusion else "(none stated)"

    user_message = f"""Patient vignette:
{patient_text}

Trial: {trial_title}

Inclusion criteria:
{incl_text}

Exclusion criteria:
{excl_text}

Determine eligibility for this specific patient."""

    system = SYSTEM_PROMPT
    if strategy == "few_shot":
        system = SYSTEM_PROMPT + "\nHere are two worked examples:\n" + FEW_SHOT_EXAMPLES

    return system, user_message


JUDGMENT_TOOL = {
    "name": "submit_eligibility_judgment",
    "description": "Submit a structured eligibility judgment for this patient-trial pair.",
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": ["eligible", "excluded", "insufficient_information"],
            },
            "cited_criterion": {
                "type": "string",
                "description": "The single specific criterion (quoted or closely paraphrased) "
                                "that most influenced this judgment.",
            },
            "rationale": {
                "type": "string",
                "description": "1-3 sentence explanation, grounded in the cited criterion.",
            },
        },
        "required": ["label", "cited_criterion", "rationale"],
    },
}
