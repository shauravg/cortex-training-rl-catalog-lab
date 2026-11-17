#!/usr/bin/env python
"""Error analysis: same listing, before vs after training.

Reads eval_completions.jsonl written by evaluate.py and prints listings the
first run got wrong and the second got right (and the reverse), plus the
violation confusion matrix for each run.

  python analyze.py --a v2/base --b v2/rl_s42 --n 4
"""

import argparse
import json
from collections import Counter, defaultdict

from listing_task import VIOLATIONS, parse

ap = argparse.ArgumentParser()
ap.add_argument("--completions", default="../eval_completions.jsonl")
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--n", type=int, default=4)
args = ap.parse_args()

runs = defaultdict(dict)
for line in open(args.completions):
    r = json.loads(line)
    runs[r["run"]][r["listing_id"]] = r


def verdict(r):
    p = parse(r["output"])
    return p["violation"] if p else "INVALID"


for name in (args.a, args.b):
    conf = Counter((r["label"]["violation"], verdict(r)) for r in runs[name].values())
    print(f"\n== {name}: gold -> predicted")
    for g in VIOLATIONS:
        row = {p: conf[(g, p)] for p in VIOLATIONS + ["INVALID"] if conf[(g, p)]}
        print(f"  {g:22s} {row}")

shared = runs[args.a].keys() & runs[args.b].keys()
fixed = [i for i in shared if verdict(runs[args.a][i]) != runs[args.a][i]["label"]["violation"]
         and verdict(runs[args.b][i]) == runs[args.b][i]["label"]["violation"]]
broke = [i for i in shared if verdict(runs[args.a][i]) == runs[args.a][i]["label"]["violation"]
         and verdict(runs[args.b][i]) != runs[args.b][i]["label"]["violation"]]
print(f"\nfixed by {args.b}: {len(fixed)}   broken by {args.b}: {len(broke)}   of {len(shared)}")

for title, ids in (("FIXED", fixed), ("BROKEN", broke)):
    # one example per gold violation type first, then fill
    seen, picked = set(), []
    for i in sorted(ids):
        v = runs[args.a][i]["label"]["violation"]
        if v not in seen:
            seen.add(v); picked.append(i)
    picked += [i for i in sorted(ids) if i not in picked]
    for i in picked[: args.n]:
        a, b = runs[args.a][i], runs[args.b][i]
        print(f"\n--- {title} listing {i} (template {a['template_id']}) gold={a['label']}")
        print(f"  text : {a['description'][:220]}")
        print(f"  {args.a:12s}: {a['output'][:200]}")
        print(f"  {args.b:12s}: {b['output'][:200]}")
