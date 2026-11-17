#!/usr/bin/env python
"""Build the SFT baseline dataset from the same governed, redacted listings.

SFT gets exactly what RL gets: the same prompt and the same gold labels.
The only extra thing SFT needs is a full target answer to copy, so the
"reason" field uses one fixed sentence per violation type.

  python build_sft_data.py ../data/listings_v2_train.jsonl \
      ../cortex-training/recipes/sft_quickstart/conversational/data/listings_v2_sft.jsonl
"""

import json
import sys

from listing_task import build_prompt

REASONS = {
    "none": "No policy issue: no off-platform contact, counterfeit signal, or prohibited item.",
    "off_platform_contact": "The seller asks buyers to contact or pay them outside the platform.",
    "counterfeit_signal": "The description signals a replica, copy, or non-authentic item.",
    "prohibited_item": "The listing includes an item the marketplace does not allow.",
}

src, dst = sys.argv[1], sys.argv[2]
n = 0
with open(src) as f, open(dst, "w") as out:
    for line in f:
        row = json.loads(line)
        gold = row["label"]
        answer = {"category": gold["category"], "brand": gold["brand"],
                  "violation": gold["violation"], "reason": REASONS[gold["violation"]]}
        messages = build_prompt(row) + [{"role": "assistant", "content": json.dumps(answer)}]
        out.write(json.dumps({"messages": messages}) + "\n")
        n += 1
print(f"wrote {n} SFT conversations to {dst}")
