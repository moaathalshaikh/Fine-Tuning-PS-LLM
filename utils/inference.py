"""
Inference utilities for Psychological Safety classification.
"""

import json
import re
import torch

# ==========================================================
# Category Normalization
# ==========================================================

CATEGORY_MAP = {

    "Recommendations": "Recommending changes",
    "Recommendation": "Recommending changes",

    "Expressing concern": "Expressing concerns",
    "Concern": "Expressing concerns",

    "Seeking Help": "Seeking help",

    "Sharing Negative Feedback": "Sharing negative feedback",

    "Drawing Attention to Errors": "Drawing attention to errors",

    "Admitting Mistake": "Admitting mistakes",

    "Disagreeing with suggestions": "Disagreeing with suggestions/ideas",
    "Disagreeing with ideas": "Disagreeing with suggestions/ideas",

}


# ==========================================================
# Prompt Builder
# ==========================================================

def build_prompt(
    prompt_template: str,
    quote_id: str,
    quote_text: str,
) -> str:
    """
    Build the final prompt by appending one quote.
    """

    return (
        prompt_template
        + "\n\n"
        + "Quotes:\n\n"
        + f"id_quote: {quote_id}\n"
        + f"related_quote: {quote_text}\n"
    )


# ==========================================================
# Model Inference
# ==========================================================

def predict_quote(
    model,
    tokenizer,
    prompt_template: str,
    quote_id: str,
    quote_text: str,
    max_new_tokens: int = 96,
):
    """
    Run inference for a single quote.
    """

    prompt = build_prompt(
        prompt_template,
        quote_id,
        quote_text,
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    return response.strip()


# ==========================================================
# Response Parsing
# ==========================================================

def parse_response(response: str):
    """
    Parse model output and normalize category names.

    Supports:

        [{...}]
        {...}

    Also supports truncated JSON responses by extracting
    the category with a regular expression.
    """

    response = (
        response
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:

        # --------------------------------------------------
        # Try JSON object
        # --------------------------------------------------

        if response.startswith("{"):

            end = response.rfind("}")

            if end == -1:
                raise ValueError

            response_json = response[: end + 1]

            data = json.loads(response_json)

        # --------------------------------------------------
        # Otherwise expect JSON array
        # --------------------------------------------------

        else:

            start = response.find("[")
            end = response.rfind("]")

            if start == -1 or end == -1:
                raise ValueError

            response_json = response[start : end + 1]

            data = json.loads(response_json)

            if isinstance(data, list):
                data = data[0]

    except Exception:

        # ----------------------------------------------
        # Fallback:
        # Extract category even if JSON is incomplete.
        # ----------------------------------------------

        match = re.search(
            r'"category"\s*:\s*"([^"]+)"',
            response,
        )

        if match:

            category = match.group(1).strip()

            category = CATEGORY_MAP.get(
                category,
                category,
            )

            return {
                "category": category,
            }

        raise ValueError("Unable to parse model response.")

    # --------------------------------------------------
    # Normalize category
    # --------------------------------------------------

    if "category" in data:

        category = data["category"].strip()

        data["category"] = CATEGORY_MAP.get(
            category,
            category,
        )

    return data
    # ------------------------------------------------------
    # Try JSON object first
    # ------------------------------------------------------

    if response.startswith("{"):

        end = response.rfind("}")

        if end == -1:
            raise ValueError("Invalid JSON object.")

        response = response[: end + 1]

        data = json.loads(response)

    # ------------------------------------------------------
    # Otherwise expect JSON array
    # ------------------------------------------------------

    else:

        start = response.find("[")
        end = response.rfind("]")

        if start == -1 or end == -1:
            raise ValueError("JSON array not found.")

        response = response[start : end + 1]

        data = json.loads(response)

        if not isinstance(data, list):
            raise ValueError("Expected a JSON array.")

        if len(data) == 0:
            raise ValueError("Empty JSON array.")

        data = data[0]

    # ------------------------------------------------------
    # Normalize category names
    # ------------------------------------------------------

    if "category" in data:

        category = data["category"].strip()

        data["category"] = CATEGORY_MAP.get(
            category,
            category,
        )

    return data