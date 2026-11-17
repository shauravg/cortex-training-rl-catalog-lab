#!/usr/bin/env python
"""Charts: reward curves and held-out results. Writes PNGs to ./images/."""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "images"
OUT.mkdir(exist_ok=True)
BLUE, DARK, ORANGE, GREEN, GREY, PURPLE = "#29b5e8", "#11567f", "#ff9f36", "#2e9d5b", "#8a99a8", "#7d44cf"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False})


def rewards(log: Path) -> list[float]:
    return [float(x) for x in re.findall(r"'reward': '([-0-9.e]+)'", log.read_text())]


def smooth(xs, w=10):
    return [sum(xs[max(0, i - w + 1): i + 1]) / len(xs[max(0, i - w + 1): i + 1]) for i in range(len(xs))]


# --- reward curves ---------------------------------------------------------
curves = [("v1 data (shared templates)", ROOT / "recipe" / "full.log", GREY, "--"),
          ("v2 data, seed 42", ROOT / "recipe" / "v2_s42.log", DARK, "-"),
          ("v2 data, seed 7", ROOT / "recipe" / "v2_s7.log", BLUE, "-")]
fig, ax = plt.subplots(figsize=(9, 4.2))
for label, log, color, style in curves:
    if log.exists() and (r := rewards(log)):
        ax.plot(range(1, len(r) + 1), smooth(r), color=color, linestyle=style, linewidth=2.2, label=label)
ax.axhline(1.0, color="#c9d3dc", linewidth=1)
ax.set_xlabel("GRPO step"); ax.set_ylabel("training reward (10-step mean)")
ax.set_ylim(0, 1.05); ax.legend(frameon=False, loc="lower right")
ax.set_title("Training reward climbs, and saturates fastest on the easy v1 data", loc="left", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "04-reward.png", dpi=200); print(OUT / "04-reward.png")

# --- held-out results ------------------------------------------------------
res = json.loads((ROOT / "eval_results.json").read_text())
order = [("base", "Qwen3-1.7B base", GREY), ("base+fewshot", "base + 4-shot prompt", PURPLE),
         ("sft", "SFT", ORANGE), ("rl_s42", "GRPO seed 42", DARK), ("rl_s7", "GRPO seed 7", BLUE),
         ("frontier:claude-sonnet-4-5", "Claude Sonnet 4.5", GREEN)]
present = [(k, n, c) for k, n, c in order if f"v2/{k}" in res]
metrics = [("missed_violation_rate", "missed violations"), ("false_alarm_rate", "false alarms on clean listings")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, (m, title) in zip(axes, metrics):
    vals = [res[f"v2/{k}"][m] * 100 for k, _, _ in present]
    bars = ax.barh([n for _, n, _ in present], vals, color=[c for _, _, c in present])
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=11)
    ax.set_title(title, loc="left", fontsize=13); ax.set_xlim(0, max(vals + [10]) * 1.25)
    ax.set_xlabel("% (lower is better)")
axes[0].invert_yaxis()  # shared y: invert once, or the two calls cancel out
fig.suptitle("Held-out v2 listings: phrasings never seen in training", x=0.01, ha="left", fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "05-results.png", dpi=200); print(OUT / "05-results.png")
