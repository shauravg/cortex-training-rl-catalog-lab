#!/usr/bin/env python
"""Step 5: evaluate models on held-out listings.

Each --model is NAME=DEPLOYMENT_ID on the Cortex Training inference endpoint.
The frontier baseline runs through Snowflake AI_COMPLETE, so test data never
leaves the account. --fewshot adds 4 worked examples from the TRAIN split to
the prompt, a cheap baseline to compare RL against.

  python evaluate.py --test ../data/listings_v2_test.jsonl --n 400 --tag v2 \
      --url $URL --model base=sf-ct-... --model rl=sf-ct-... \
      --frontier claude-sonnet-4-5 --fewshot-for base
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from listing_task import VIOLATIONS, build_prompt, parse, score

FEWSHOT_TRAIN = os.path.join(os.path.dirname(__file__), "..", "data", "listings_v2_train.jsonl")


def load(path: str, n: int, seed: int = 0) -> list[dict]:
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    random.Random(seed).shuffle(rows)  # stratify-ish: avoid head-of-file bias
    return rows[:n]


def fewshot_examples(path: str = FEWSHOT_TRAIN) -> list[dict]:
    """One training example per violation type, with its gold answer."""
    picked = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            picked.setdefault(row["label"]["violation"], row)
            if len(picked) == len(VIOLATIONS):
                break
    msgs = []
    for row in picked.values():
        g = row["label"]
        msgs += [build_prompt(row)[1],
                 {"role": "assistant", "content": json.dumps({**g, "reason": "example"})}]
    return msgs


def messages_for(row: dict, fewshot: list[dict] | None) -> list[dict]:
    system, user = build_prompt(row)
    return [system] + (fewshot or []) + [user]


def endpoint_predict(url: str, deployment: str, rows: list[dict], fewshot=None) -> list[str]:
    client = OpenAI(base_url=url, api_key=os.environ["ARCTIC_CORTEX_PAT"])

    def one(row):
        resp = client.chat.completions.create(
            model=deployment, messages=messages_for(row, fewshot), temperature=0.0,
            max_completion_tokens=256,
        )
        return resp.choices[0].message.content

    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(one, rows))


def frontier_predict(connection: str, model: str, rows: list[dict], warehouse: str) -> list[str]:
    import snowflake.connector

    conn = snowflake.connector.connect(connection_name=connection, warehouse=warehouse)

    def one(row):
        msgs = build_prompt(row)
        prompt = msgs[0]["content"] + "\n\n" + msgs[1]["content"]
        cur = conn.cursor()
        cur.execute("SELECT AI_COMPLETE(%s, %s)", (model, prompt))
        text = cur.fetchone()[0]
        # The connector returns AI_COMPLETE's string JSON-encoded: '"```json\\n{...}"'
        return json.loads(text) if text.startswith('"') else text

    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(one, rows))
    conn.close()
    return out


def metrics(texts: list[str], rows: list[dict]) -> dict:
    preds = [parse(t) for t in texts]
    golds = [r["label"] for r in rows]
    bad = [g["violation"] != "none" for g in golds]
    n_bad, n_clean = sum(bad), len(bad) - sum(bad)
    missed = sum(1 for p, b in zip(preds, bad) if b and (p is None or p["violation"] == "none"))
    false_alarm = sum(1 for p, b in zip(preds, bad) if not b and p and p["violation"] != "none")
    by_type = {}
    for v in VIOLATIONS:
        idx = [i for i, g in enumerate(golds) if g["violation"] == v]
        if idx:
            by_type[v] = round(sum(bool(preds[i]) and preds[i]["violation"] == v for i in idx) / len(idx), 3)
    return {
        "n": len(rows),
        "reward": round(sum(score(p, g) for p, g in zip(preds, golds)) / len(rows), 3),
        "valid_json": round(sum(p is not None for p in preds) / len(rows), 3),
        "category_acc": round(sum(bool(p) and p["category"] == g["category"] for p, g in zip(preds, golds)) / len(rows), 3),
        "brand_acc": round(sum(bool(p) and str(p.get("brand", "")).lower() == g["brand"].lower() for p, g in zip(preds, golds)) / len(rows), 3),
        "violation_acc": round(sum(bool(p) and p["violation"] == g["violation"] for p, g in zip(preds, golds)) / len(rows), 3),
        "missed_violation_rate": round(missed / max(n_bad, 1), 3),
        "false_alarm_rate": round(false_alarm / max(n_clean, 1), 3),
        "recall_by_type": by_type,
        "top_category_errors": Counter(
            f'{g["category"]}->{p["category"] if p else "INVALID"}'
            for p, g in zip(preds, golds) if not p or p["category"] != g["category"]).most_common(3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="../data/listings_v2_test.jsonl")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tag", default="v2", help="prefix for result keys, e.g. v1 or v2")
    ap.add_argument("--url", help="Cortex Training inference base URL")
    ap.add_argument("--model", action="append", default=[], help="NAME=DEPLOYMENT_ID, repeatable")
    ap.add_argument("--fewshot-for", action="append", default=[], help="also run NAME with few-shot prompt")
    ap.add_argument("--frontier", default="", help="AI_COMPLETE model name; empty to skip")
    ap.add_argument("--connection", default="cortex-training")
    ap.add_argument("--warehouse", default="COMPUTE_WH")
    ap.add_argument("--out", default="../eval_results.json", help="merged with existing results")
    ap.add_argument("--completions", default="../eval_completions.jsonl", help="raw outputs appended here")
    args = ap.parse_args()

    rows = load(args.test, args.n)
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    runs = []
    for spec in args.model:
        name, dep = spec.split("=", 1)
        runs.append((name, lambda d=dep: endpoint_predict(args.url, d, rows)))
        if name in args.fewshot_for:
            shots = fewshot_examples()
            runs.append((f"{name}+fewshot", lambda d=dep, s=shots: endpoint_predict(args.url, d, rows, s)))
    if args.frontier:
        runs.append((f"frontier:{args.frontier}",
                     lambda: frontier_predict(args.connection, args.frontier, rows, args.warehouse)))

    with open(args.completions, "a") as comp:
        for name, fn in runs:
            texts = fn()
            key = f"{args.tag}/{name}"
            results[key] = metrics(texts, rows)
            print(key, json.dumps(results[key]), flush=True)
            for row, text in zip(rows, texts):
                comp.write(json.dumps({"run": key, "listing_id": row["listing_id"],
                                       "template_id": row.get("template_id"), "label": row["label"],
                                       "title": row["title"], "description": row["description"],
                                       "output": text}) + "\n")
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
