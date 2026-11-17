"""Prompt + verifiable reward for marketplace listing review.

The reward is the business policy: a missed violation costs far more than a
false alarm, so it is weighted that way. TRL calls reward functions with the
completions plus every dataset column as a keyword argument.
"""

from __future__ import annotations

import json
import re

CATEGORIES = ["electronics", "apparel", "home_kitchen", "toys", "beauty"]
VIOLATIONS = ["none", "off_platform_contact", "counterfeit_signal", "prohibited_item"]

SYSTEM = (
    "You review marketplace listings. PII in the text has already been replaced "
    "with placeholders like [EMAIL] or [PHONE_NUMBER]. Return only JSON: "
    '{"category": one of ' + json.dumps(CATEGORIES) + ', "brand": string, '
    '"violation": one of ' + json.dumps(VIOLATIONS) + ', "reason": short string}'
)

# Reward weights. Tune to your own risk appetite.
W_CATEGORY = 0.25
W_BRAND = 0.15
W_VIOLATION = 0.60
MISSED_VIOLATION_PENALTY = -1.0   # said "none" when it was a violation
FALSE_ALARM_PENALTY = -0.2        # flagged a clean listing
INVALID_JSON_REWARD = -1.0       # never cheaper than a missed violation


def build_prompt(row: dict) -> list[dict]:
    user = f"Title: {row['title']}\nPrice: ${row['price']}\nDescription: {row['description']}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        out = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if out.get("category") not in CATEGORIES or out.get("violation") not in VIOLATIONS:
        return None
    return out


def score(pred: dict | None, gold: dict) -> float:
    if pred is None:
        return INVALID_JSON_REWARD
    r = 0.0
    r += W_CATEGORY * (pred["category"] == gold["category"])
    r += W_BRAND * (str(pred.get("brand", "")).strip().lower() == gold["brand"].lower())
    if pred["violation"] == gold["violation"]:
        r += W_VIOLATION
    elif gold["violation"] != "none" and pred["violation"] == "none":
        r += MISSED_VIOLATION_PENALTY
    elif gold["violation"] == "none":
        r += FALSE_ALARM_PENALTY
    # wrong violation type but still flagged: no bonus, no penalty
    return r


def _text(completion) -> str:
    # Conversational completions arrive as [{"role": "assistant", "content": ...}]
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion


def listing_reward(completions, label, **kwargs) -> list[float]:
    """TRL reward function: one float per completion."""
    return [score(parse(_text(c)), gold) for c, gold in zip(completions, label)]
