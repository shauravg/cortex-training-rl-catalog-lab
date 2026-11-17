#!/usr/bin/env python
"""TRL async-GRPO on Cortex Training for marketplace listing review.

Reuses the Arctic-Platform recipe unchanged:
  recipes/rl/trl/gsm8k_cortex/run_qwen3_1.7b_gsm8k_grpo_cortex.py
and swaps three things: the dataset, the prompt format, and the reward.

It also saves a weights-only checkpoint before the recipe shuts the client
down (the recipe itself saves nothing), and writes the job and checkpoint IDs
into deploy_tuned.json for the model deployment step.

Copy this file into Arctic-Platform/recipes/rl/trl/gsm8k_cortex/ next to
listing_task.py and run_listing.sh, then: DATA=/abs/path/train.jsonl bash run_listing.sh
"""

from __future__ import annotations

import json
import os

import datasets
import trl.rewards

import run_qwen3_1_7b_gsm8k_grpo_cortex as recipe  # see README: symlink with underscores
from listing_task import build_prompt, listing_reward

DATA = os.environ.get("DATA", "data/listings_train.jsonl")
CHECKPOINT_ID = os.environ.get("CHECKPOINT_ID", "listing-grpo-final")
DEPLOY_SPEC = os.environ.get("DEPLOY_SPEC", "deploy_tuned.json")

_load_dataset = datasets.load_dataset
_build_client = recipe.build_client


def load_listings(*_args, split="train", **_kwargs):
    ds = _load_dataset("json", data_files=DATA, split="train")
    if split.startswith("train[:"):
        ds = ds.select(range(int(split[len("train[:"):-1])))
    return ds


def format_listing(row: dict) -> dict:
    return {"prompt": build_prompt(row), "label": row["label"]}


def build_client_with_save(*args, **kwargs):
    """Save weights-only before main()'s finally block calls client.shutdown()."""
    client = _build_client(*args, **kwargs)
    shutdown = client.shutdown

    def save_then_shutdown():
        try:
            out = client.save_checkpoint(checkpoint_id=CHECKPOINT_ID, checkpoint_type="weights-only")
            job_id = str(client.jobs.training).split(":")[0]  # "<uuid>:training:0" -> parent job UUID
            print(f"[listing] saved {CHECKPOINT_ID} on training job {job_id}: {out}", flush=True)
            with open(DEPLOY_SPEC) as f:
                spec = json.load(f)
            spec["source"].update(job_id=str(job_id), checkpoint_id=CHECKPOINT_ID)
            with open(DEPLOY_SPEC, "w") as f:
                json.dump(spec, f, indent=2)
            print(f"[listing] wrote {DEPLOY_SPEC}", flush=True)
        except Exception as e:
            print(f"[listing] checkpoint save failed: {e}", flush=True)
        shutdown()

    client.shutdown = save_then_shutdown
    return client


# The recipe imports these inside main(), so patching the modules is enough.
datasets.load_dataset = load_listings
trl.rewards.accuracy_reward = listing_reward
recipe.format_sample = format_listing
recipe.build_client = build_client_with_save

if __name__ == "__main__":
    recipe.main()
