#!/usr/bin/env python
"""ROI worksheet. Every input is an assumption you replace with your own numbers.

  python roi.py --eval eval_results.json --listings-per-day 50000 \
      --reviewer-cost-per-listing 0.40 --missed-violation-cost 25 \
      --frontier-cost-per-1k 20 --tuned-cost-per-1k 0.50
"""

import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--eval", default="../eval_results.json")
ap.add_argument("--tag", default="v2", help="only rows from this eval set")
ap.add_argument("--listings-per-day", type=float, default=50_000)
ap.add_argument("--violation-rate", type=float, default=0.30)
ap.add_argument("--reviewer-cost-per-listing", type=float, default=0.40)
ap.add_argument("--missed-violation-cost", type=float, default=25.0,
                help="Chargebacks, trust & safety escalations, brand risk per missed bad listing")
ap.add_argument("--false-alarm-cost", type=float, default=0.40,
                help="A false alarm goes to a human reviewer")
ap.add_argument("--frontier-cost-per-1k", type=float, default=20.0)
ap.add_argument("--tuned-cost-per-1k", type=float, default=0.50)
args = ap.parse_args()

results = json.load(open(args.eval))
per_day = args.listings_per_day
bad, clean = per_day * args.violation_rate, per_day * (1 - args.violation_rate)
print(f"Human-only review: ${per_day * args.reviewer_cost_per_listing * 365:,.0f}/yr\n")

for name, m in results.items():
    if not name.startswith(f"{args.tag}/"):
        continue
    inference = args.frontier_cost_per_1k if "frontier" in name else args.tuned_cost_per_1k
    daily = (per_day / 1000 * inference
             + bad * m["missed_violation_rate"] * args.missed_violation_cost
             + clean * m["false_alarm_rate"] * args.false_alarm_cost)
    print(f"{name:32s} ${daily * 365:>14,.0f}/yr  "
          f"(missed {m['missed_violation_rate']:.1%}, false alarm {m['false_alarm_rate']:.1%})")
